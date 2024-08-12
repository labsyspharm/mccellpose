# ---------------------------------------------------------------------------- #
#                              Process large image                             #
# ---------------------------------------------------------------------------- #
import argparse
import datetime
import pathlib
import sys
import time

import cellpose.denoise
import dask.array as da
import dask.diagnostics
import dask_image.ndmeasure
import palom
import scipy.ndimage as ndi
import skimage.exposure
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

    if np.all(tmask == 0):
        return tmask.astype("bool")

    struct_elem = ndi.generate_binary_structure(tmask.ndim, 1)
    contour = ndi.grey_dilation(tmask, footprint=struct_elem) != ndi.grey_erosion(
        tmask, footprint=struct_elem
    )
    return (tmask > 0) & ~contour


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
        nargs=1,
        type=int,
        required=True,
        help='DNA channel to segment (1-based)',
    )
    args = parser.parse_args()

    args.output = pathlib.Path(args.output)
    if not args.output.parent.exists():
        print(
            f"ERROR: output path parent directoy does not exsit: {args.output.parent}",
            file=sys.stderr,
        )
        sys.exit(1)

    start = int(time.perf_counter())

    reader = palom.reader.OmePyramidReader(args.input)
    img = reader.pyramid[0][0]
    dn_model = cellpose.denoise.CellposeDenoiseModel(
        # this seems to be the best atm
        gpu=True,
        model_type="cyto3",
        restore_type="deblur_cyto3",
    )

    _binary_mask = img.map_overlap(
        segment_tile,
        depth={0: 128, 1: 128},
        boundary="none",
        dtype=bool,
        dn_model=dn_model,
    )
    _binary_mask = da_to_zarr(_binary_mask, num_workers=6)
    binary_mask = da.from_zarr(_binary_mask)

    end = int(time.perf_counter())
    print("\nelapsed (cellpose):", datetime.timedelta(seconds=end - start))

    _labeled_mask = dask_image.ndmeasure.label(binary_mask)[0]
    labeled_mask = da_to_zarr(_labeled_mask)

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
