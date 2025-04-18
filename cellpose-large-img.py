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

import cellpose.models
import dask.array
import dask.config
import ome_types
import skimage.exposure
import skimage.segmentation
import sklearn.mixture
import threadpoolctl
import tifffile
import tqdm
import zarr
import numpy as np


def segment_tile(timg, dn_model, intensity_max, cytoplasm_thickness):
    timg = skimage.exposure.rescale_intensity(
        timg, in_range=(0, intensity_max), out_range="float"
    )
    labels_nucleus = dn_model.eval(
        timg,
        diameter=17.0,
        normalize=False,
    )[0]
    labels_cell = skimage.segmentation.expand_labels(
        labels_nucleus, cytoplasm_thickness
    )

    return labels_nucleus, labels_cell


def auto_threshold(img):

    assert img.ndim == 2

    ys, xs = (slice(0, s, np.ceil(s / 200).astype(int)) for s in img.shape)
    img = img[ys, xs]
    img_log, img_max = dask.compute(np.log(img[img > 0]), img.max())
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    gmm.fit(img_log.reshape((-1,1)))
    means = gmm.means_[:, 0]
    _, _, i = np.argsort(means)
    mean = means[i]
    std = gmm.covariances_[i, 0, 0] ** 0.5

    lmax = mean + 2 * std
    vmax = min(np.exp(lmax), img_max)

    return vmax


def pad_block(zarray, c):
    """Return block c from zarray, padded to the full chunk size"""
    img = zarray.blocks[c]
    diff = np.subtract(zarray.chunks, img.shape)
    if any(diff > 0):
        pad_width = np.vstack([np.zeros(len(diff), int), diff]).T
        img = np.pad(img, pad_width)
    return img


def get_low_res(reader):
    """Return a low resolution pyramid level, at least 200x200 px for auto_threshold"""

    for img in reversed(reader.pyramid):
        if all(s >= 200 for s in img.shape[1:3]):
            return img
    return reader.pyramid[0]


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
        '--pixel-size',
        type=float,
        help="Pixel size (nominal image resolution) in microns. You may omit"
        " this if your input OME-TIFF contains accurate pixel size metadata.",
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Enable GPU-based processing for CellPose (default: no, use CPU)'
    )
    parser.add_argument(
        '--output-discard',
        help='Discard mask output image',
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

    threadpoolctl.threadpool_limits(1)
    pool = concurrent.futures.ThreadPoolExecutor(args.jobs)
    if args.use_gpu:
        dask.config.set(pool=pool)

    tiff = tifffile.TiffFile(args.input)
    ome = ome_types.from_xml(tiff.ome_metadata)
    if args.pixel_size:
        pixel_size = args.pixel_size
    else:
        pixel_size = ome.images[0].pixels.physical_size_x_quantity.to("micron").m
        print(f"Pixel size detected from OME-TIFF: {pixel_size} μm")
    img = zarr.open(tiff.series[0][args.channel - 1].aszarr(level=0))
    expand_size_px = round(args.expand_size / pixel_size)
    intensity_max = auto_threshold(dask.array.from_zarr(img))
    dn_model = cellpose.models.CellposeModel(gpu=args.use_gpu, model_type='nuclei')

    tw = args.tile_width
    overlap = round(args.tile_overlap / pixel_size)

    step = tw - overlap
    ys = np.arange(0, img.shape[0], step)
    xs = np.arange(0, img.shape[1], step)
    labels_full = zarr.open(
        'temp_labels.zarr',
        mode='w',
        shape=(2,) + img.shape,
        chunks=(1, tw, tw),
        dtype=np.uint32,
    )
    mask_discard = zarr.open(
        'temp_discard.zarr',
        mode='w',
        shape=img.shape,
        chunks=(tw, tw),
        dtype=bool,
    )
    num_masks = 0

    def get_tile(arr, y, x):
        return arr[y : y + tw, x : x + tw]

    def work(y, x):
        return segment_tile(
            get_tile(img, y, x), dn_model, intensity_max, expand_size_px
        )

    futures = {
        pool.submit(work, y, x): (y, x)
        for y, x in itertools.product(ys, xs)
    }
    f_iter = concurrent.futures.as_completed(futures)
    for f in tqdm.tqdm(f_iter, total=len(futures)):
        y, x = futures.pop(f)
        labels_nucleus, labels_cell = f.result()
        # Make an in-memory copy of the slice of the zarr arrays corresponding
        # to the tile we just segmented. We will operate on the copies, writing
        # them back to the zarr array after processing all cells in this tile.
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
        # Write working copies back to the zarr arrays.
        labels_full[:, y : y + tw, x : x + tw] = lf_window
        mask_discard[y : y + tw, x : x + tw] = md_window

    end = int(time.perf_counter())
    print("\nelapsed (cellpose):", datetime.timedelta(seconds=end - start))

    large_objects = 0
    for y, x in tqdm.tqdm(list(itertools.product(ys, xs))):
        dtile = get_tile(mask_discard, y, x)
        dtile = skimage.morphology.remove_small_objects(dtile, 2)
        dlabels = skimage.measure.label(dtile)
        for p in skimage.measure.regionprops(dlabels):
            oh = p.bbox[2] - p.bbox[0]
            ow = p.bbox[3] - p.bbox[1]
            if (
                ((p.bbox[0] == 0 or p.bbox[2] == dtile.shape[0]) and oh >= overlap)
                or ((p.bbox[1] == 0 or p.bbox[3] == dtile.shape[1]) and ow >= overlap)
            ):
                large_objects += 1
    large_objects = round(large_objects / 2)
    if large_objects:
        print(
            f"WARNING: Found {large_objects} large cells spanning an entire tile overlap"
            f" that could not be segmented"
        )

    block_coords = (itertools.product(*(range(s) for s in labels_full.cdata_shape)))
    tifffile.imwrite(
        args.output,
        (pad_block(labels_full, c) for c in block_coords),
        shape=labels_full.shape,
        dtype=labels_full.dtype,
        tile=(tw, tw),
        compression="zlib",
        predictor=True,
        maxworkers=1,
    )

    if args.output_discard:
        block_coords = (itertools.product(*(range(s) for s in mask_discard.cdata_shape)))
        tifffile.imwrite(
            args.output_discard,
            (pad_block(mask_discard, c) for c in block_coords),
            shape=mask_discard.shape,
            dtype=mask_discard.dtype,
            tile=(tw, tw),
            compression="zlib",
            maxworkers=1,
        )

    end = int(time.perf_counter())
    print("\nelapsed (total):", datetime.timedelta(seconds=end - start))


if __name__ == '__main__':
    main()
