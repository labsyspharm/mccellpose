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
import numcodecs
import palom
import skimage.exposure
import skimage.segmentation
import tifffile
import torch.cuda
import tqdm
import zarr
import numpy as np


def segment_tile(timg, dn_model):
    # timg = skimage.exposure.adjust_gamma(
    #     skimage.exposure.rescale_intensity(
    #         timg, in_range=(500, 40_000), out_range="float"
    #     ),
    #     0.7,
    # )
    timg = skimage.exposure.rescale_intensity(
        timg, in_range=(500, 40_000), out_range="float"
    )
    tmask = dn_model.eval(
        timg,
        diameter=15.0,
        channels=[0, 0],
        # inputs are globally normalized already
        normalize=False,
        flow_threshold=0,
        # GPU with 8 GB of RAM can handle 1024x1024 images
        tile=True,
    )[0]

    return tmask


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
    dn_model = cellpose.denoise.CellposeDenoiseModel(
        # this seems to be the best atm
        gpu=args.use_gpu,
        model_type='cyto3',
        restore_type='deblur_cyto3',
    )

    tw = 2048
    overlap = 128

    step = tw - overlap
    ys = np.arange(0, img.shape[0], step)
    xs = np.arange(0, img.shape[1], step)
    labels_full = zarr.open(
        'temp.zarr',
        mode='w',
        shape=img.shape,
        chunks=(tw, tw),
        dtype=np.uint32,
        compressor=numcodecs.Zstd(),
    )
    n_masks = 0

    def work(y, x):
        return segment_tile(img[y : y + tw, x : x + tw], dn_model)

    pool = concurrent.futures.ThreadPoolExecutor(args.jobs)
    futures = {
        (y, x): pool.submit(work, y, x)
        for y, x in itertools.product(ys, xs)
    }
    for (y, x), f in tqdm.tqdm(futures.items()):
        labels = f.result()
        lf_view = labels_full[y : y + tw, x : x + tw]
        lh, lw = labels.shape
        for prop in skimage.measure.regionprops(labels):
            bb = prop.bbox
            if bb[0] == 0 or bb[1] == 0 or bb[2] == lh or bb[3] == lw:
                continue
            if lf_view[prop.slice][prop.image].any():
                continue
            n_masks += 1
            lf_view[prop.slice][prop.image] = n_masks
        labels_full[y : y + tw, x : x + tw] = lf_view

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
