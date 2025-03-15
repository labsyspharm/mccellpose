# ---------------------------------------------------------------------------- #
#                              Process large image                             #
# ---------------------------------------------------------------------------- #
import argparse
import concurrent.futures
import datetime
import gc
import itertools
import pathlib
import sys
import time

import cellpose.denoise
import dask.array as da
import dask.diagnostics
import dask_image.ndmeasure as dim
import numcodecs
import ome_types
import palom
import skimage.exposure
import skimage.segmentation
import tifffile
import torch.cuda
import tqdm
import zarr
import numpy as np


def segment_tile(timg, dn_model, cytoplasm_thickness):
    # timg = skimage.exposure.adjust_gamma(
    #     skimage.exposure.rescale_intensity(
    #         timg, in_range=(500, 40_000), out_range="float"
    #     ),
    #     0.7,
    # )
    timg = skimage.exposure.rescale_intensity(
        timg, in_range=(500, 40_000), out_range="float"
    )
    labels_nucleus = dn_model.eval(
        timg,
        diameter=15.0,
        channels=[0, 0],
        # inputs are globally normalized already
        normalize=False,
    )[0]
    labels_cell = skimage.segmentation.expand_labels(
        labels_nucleus, cytoplasm_thickness
    )

    return labels_nucleus, labels_cell


def da_to_zarr(da_img, zarr_store=None, num_workers=None, out_shape=None, chunks=None):
    if zarr_store is None:
        if out_shape is None:
            out_shape = da_img.shape
        if chunks is None:
            chunks = da_img.chunksize
        zarr_store = zarr.create(
            out_shape, chunks=chunks, dtype=da_img.dtype, overwrite=True
        )
    with dask.diagnostics.ProgressBar():
        da_img.to_zarr(zarr_store, compute=False).compute(num_workers=2)
    return zarr_store


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', required=True, help='Input image'
    )
    parser.add_argument(
        '-o', '--output', required=True, help='Output image'
    )
    parser.add_argument(
        '-c', '--channel',
        type=int,
        required=True,
        help='DNA channel to segment (1-based)',
    )
    parser.add_argument(
        '--tile-width',
        type=int,
        default=2048,
        help='Tile width in pixels',
    )
    parser.add_argument(
        '--tile-overlap',
        type=float,
        default=50.0,
        help='Tile overlap in microns',
    )
    parser.add_argument(
        '--expand-size',
        type=float,
        required=True,
        help='Number of microns to expand nuclei masks to obtain cytoplasm masks',
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Enable GPU-based processing for CellPose (default: no, use CPU)'
    )
    parser.add_argument(
        '--jobs',
        default=1,
        type=int,
        help='Number of jobs to run simultaneously when using GPU processing'
        ' (default: 1). Increase this value by 1 until your GPU reaches ~100%%'
        ' utilization. Higher values than this will only waste RAM and VRAM'
        ' without providing a speedup. CPU processing is already implicitly'
        ' parallelized and will automatically use all available CPUs.',
    )
    args = parser.parse_args()

    args.output = pathlib.Path(args.output)
    if not args.output.parent.exists():
        print(
            f"ERROR: output path parent directoy does not exsit: {args.output.parent}",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.jobs > 1 and not args.use_gpu:
        print(
            "ERROR: can't use --jobs without --use-gpu (CPU mode is already"
            " implicitly parallelized; see --help output for details)"
        )
        sys.exit(1)

    start = int(time.perf_counter())

    reader = palom.reader.OmePyramidReader(args.input)
    img = reader.pyramid[0][0]
    expand_size_px = round(args.expand_size / reader.pixel_size)
    dn_model = cellpose.denoise.CellposeDenoiseModel(
        # this seems to be the best atm
        gpu=args.use_gpu,
        model_type='cyto3',
        restore_type='deblur_cyto3',
    )

    tw = args.tile_width
    overlap = round(args.tile_overlap / reader.pixel_size)

    step = tw - overlap
    ys = np.arange(0, img.shape[0], step)
    xs = np.arange(0, img.shape[1], step)
    labels_full = zarr.open(
        'temp_labels.zarr',
        mode='w',
        shape=(2,) + img.shape,
        chunks=(2, tw, tw),
        dtype=np.uint32,
        compressor=numcodecs.Zstd(),
    )
    mask_discard = zarr.open(
        'temp_discard.zarr',
        mode='w',
        shape=img.shape,
        chunks=(tw, tw),
        dtype=bool,
        compressor=numcodecs.Zstd(),
    )
    num_masks = 0

    def work(y, x):
        return segment_tile(
            img[y : y + tw, x : x + tw], dn_model, expand_size_px
        )

    pool = concurrent.futures.ThreadPoolExecutor(args.jobs)
    futures = {
        (y, x): pool.submit(work, y, x)
        for y, x in itertools.product(ys, xs)
    }
    for (y, x), f in tqdm.tqdm(futures.items()):
        labels_nucleus, labels_cell = f.result()
        # Make an in-memory copy of the slice of the dask arrays corresponding
        # to the tile we just segmented. We will operate on the copies, writing
        # them back to the dask array after processing all cells in this tile.
        lf_window = labels_full[:, y : y + tw, x : x + tw]
        md_window = mask_discard[y : y + tw, x : x + tw]
        lh, lw = labels_nucleus.shape
        props_nucleus = skimage.measure.regionprops(labels_nucleus)
        props_cell = skimage.measure.regionprops(labels_cell)
        for pn, pc in zip(props_nucleus, props_cell):
            bb = pc.bbox
            # If object touches edge of entire image, discard.
            if (
                (y == 0 and bb[0] == 0)
                or (x == 0 and bb[1] == 0)
                or (y == ys[-1] and bb[2] == lh)
                or (x == xs[-1] and bb[3] == lw)
            ):
                continue
            # If object intersects a previously detected cell, discard.
            intersection = (lf_window[1][pc.slice] > 0) & pc.image
            if np.sum(intersection) > pc.area * 0.02:
                continue
            # If object touches edge of tile within the interior of the image,
            # add to discard mask and stop processing this object.
            if bb[0] == 0 or bb[1] == 0 or bb[2] == lh or bb[3] == lw:
                md_window[pc.slice][pc.image] = True
                continue
            # New complete cell -- add it to the label image.
            num_masks += 1
            lf_window[0][pn.slice][pn.image] = num_masks
            lf_window[1][pc.slice][pc.image] = num_masks
            # Clear discard mask for this cell since we've seen it now.
            md_window[pc.slice][pc.image] = False
        # Write working copies back to the dask arrays.
        labels_full[:, y : y + tw, x : x + tw] = lf_window
        mask_discard[y : y + tw, x : x + tw] = md_window

    # Warn if there are any objects in discard_mask that are wider than the tile
    # overlap. This is a hint that the overlap should possibly be increased.
    mask_discard = da.from_zarr(mask_discard)
    labels_discard, _ = dim.label(mask_discard)
    objects_discard = dim.find_objects(labels_discard)
    discard_count = 0
    for i, yslice, xslice in objects_discard.itertuples():
        ysize = yslice.stop - yslice.start
        xsize = xslice.stop - xslice.start
        if ysize >= overlap or xsize >= overlap:
            print(
                f"WARNING: Found some large cells spanning an entire tile overlap that"
                f" could not be segmented"
            )
            break

    end = int(time.perf_counter())
    print("\nelapsed (cellpose):", datetime.timedelta(seconds=end - start))

    labeled_mask = labels_full

    end = int(time.perf_counter())
    print("\nelapsed (label):", datetime.timedelta(seconds=end - start))

    palom.pyramid.write_pyramid(
        [da.from_zarr(labeled_mask)],
        args.output,
        pixel_size=reader.pixel_size,
        downscale_factor=2,
        compression="zlib",
        save_RAM=True,
        is_mask=True,
    )

    end = int(time.perf_counter())
    print("\nelapsed (total):", datetime.timedelta(seconds=end - start))


if __name__ == '__main__':
    main()
