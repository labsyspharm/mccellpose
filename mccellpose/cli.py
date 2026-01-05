import argparse
import concurrent.futures
import itertools
import pathlib
import sys

import cellpose.models
import dask.array
import dask.config
import logging
import ome_types
import skimage.exposure
import skimage.segmentation
import sklearn.mixture
import threadpoolctl
import tifffile
import tqdm
import zarr
import numpy as np

from . import __version__


def segment_tile(timg, cp_model, intensity_max, cytoplasm_thickness, diameter):
    timg = skimage.exposure.rescale_intensity(
        timg, in_range=(0, intensity_max), out_range="float"
    )
    labels_nucleus = cp_model.eval(
        timg,
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


class PrintLogger:

    def info(self, msg):
        print(msg)

    def warn(self, msg):
        print("WARNING:", msg)

    def error(self, msg):
        print("ERROR:", msg)


def progress(iterable, logger, **kwargs):
    if sys.stdout.isatty():
        t = tqdm.tqdm(iterable, file=sys.stdout, **kwargs)
    else:
        f = TqdmLogWrapper(logger)
        t = tqdm.tqdm(
            iterable,
            file=f,
            ncols=80,
            mininterval=60,
            ascii=False,
            **kwargs,
        )
    yield from t


class TqdmLogWrapper:

    def __init__(self, logger):
        self.logger = logger

    def write(self, s):
        # Emit bar updates (which begin with a CR) as individual messages.
        if s[0:1] == '\r':
            self.logger.info(s[1:])


def main():

    parser = argparse.ArgumentParser(
        description="Run cellpose on an OME-TIFF using overlapping tiles for"
        " memory efficiency.",
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        type=pathlib.Path,
        help='Input image',
    )
    parser.add_argument(
        '-o', '--output-cell',
        required=True,
        type=pathlib.Path,
        help='Output label image for cell segmentation masks',
    )
    parser.add_argument(
        '--output-nucleus',
        type=pathlib.Path,
        help='Output label image for nucleus segmentation masks (optional)',
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
        help='Tile overlap in microns (default: --diameter value times 5)',
    )
    parser.add_argument(
        '--expand-size',
        type=float,
        required=True,
        help='Number of microns to expand nuclei masks to obtain cytoplasm masks',
    )
    parser.add_argument(
        '--diameter',
        type=float,
        default=10,
        help='Diameter of cell nuclei in microns (default: 10)',
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
        type=pathlib.Path,
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
    parser.add_argument('--version', action='version', version=f'mccellpose {__version__}')
    args = parser.parse_args()

    if sys.stdout.isatty():
        logger = PrintLogger()
    else:
        logging.basicConfig(
            format="%(asctime)s.%(msecs)03d %(name)-20s %(levelname)-8s : %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )
        logger = logging.getLogger()

    if not args.input.exists():
        logger.error(
            f"Input image file does not exist: {args.input}"
        )
        sys.exit(1)
    if not args.output_cell.parent.exists():
        logger.error(
            f"Output cell mask parent directory does not exist: {args.output_cell.parent}"
        )
        sys.exit(1)
    if args.output_nucleus and not args.output_nucleus.parent.exists():
        logger.error(
            f"Output nucleus mask parent directory does not exist: {args.output_nucleus.parent}"
        )
        sys.exit(1)
    if args.output_discard and not args.output_discard.parent.exists():
        logger.error(
            "Output discard mask parent directory does not exist:"
            f" {args.output_discard.parent}"
        )
        sys.exit(1)
    if args.jobs > 1 and not args.use_gpu:
        logger.error(
            "Can't use --jobs without --use-gpu (CPU mode is already"
            " implicitly parallelized; see --help output for details)"
        )
        sys.exit(1)

    if args.tile_overlap is None:
        args.tile_overlap = args.diameter * 5

    threadpoolctl.threadpool_limits(1)
    pool = concurrent.futures.ThreadPoolExecutor(args.jobs)
    if args.use_gpu:
        dask.config.set(pool=pool)

    tiff = tifffile.TiffFile(args.input)
    ome = ome_types.from_xml(tiff.ome_metadata)
    if args.pixel_size:
        pixel_size = args.pixel_size
    else:
        ppsx = ome.images[0].pixels.physical_size_x_quantity
        if ppsx is None:
            logger.error(
                "Input image has no pixel size metadata; please specify --pixel-size"
            )
            sys.exit(1)
        pixel_size = ppsx.to("micron").m
        logger.info(f"Pixel size detected from OME-TIFF: {pixel_size} μm")

    tw = args.tile_width
    if tw % 16 != 0:
        logger.error("--tile-width value must be a multiple of 16")
        sys.exit(1)

    overlap = round(args.tile_overlap / pixel_size)
    logger.info(f"Tile overlap: {args.tile_overlap} μm ({overlap} px)")
    if overlap < 3:
        logger.warn(
            "Tile overlap is very small (less than 3 pixels) -- many cells are"
            " likely to be missed"
        )
    diameter = args.diameter / pixel_size
    logger.info(f"Expected nucleus diameter: {args.diameter} μm ({diameter} px)")

    logger.info("Computing image contrast...")
    img = zarr.open(tiff.series[0][args.channel - 1].aszarr(level=0), mode="r")
    expand_size_px = round(args.expand_size / pixel_size)
    intensity_max = auto_threshold(dask.array.from_zarr(img))
    logger.info(f"Rescaling intensity to auto-detected upper limit: {intensity_max}")
    cp_model = cellpose.models.CellposeModel(gpu=args.use_gpu)

    step = tw - overlap
    # Subtract 1 from image dimensions when computing the upper limit for the
    # rolling window to omit any edge windows with a width or height of 1. This
    # works around a bug in cellpose where the gradient array is squeezed to
    # eliminate some intermediate singleton dimensions and inadvertently drops
    # this real length-1 dimension in our tiles. A 1-pixel edge window would be
    # fully covered by the overlap from the previous window anyway, so skipping
    # these windows doesn't affect our results.
    # FIXME: Omit edge windows up to the full overlap size too?
    ys = np.arange(0, img.shape[0] - 1, step)
    xs = np.arange(0, img.shape[1] - 1, step)
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
            get_tile(img, y, x), cp_model, intensity_max, expand_size_px, diameter
        )

    coords = list(itertools.product(ys, xs))
    futures = {
        pool.submit(work, y, x): (y, x)
        for y, x in coords
    }
    f_iter = concurrent.futures.as_completed(futures)
    for f in progress(
        f_iter, logger, desc="Segmenting image tiles", total=len(coords)
    ):
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
    logger.info(f"Segmentation complete -- detected {num_masks} cells")

    large_objects = 0
    for y, x in progress(coords, logger, desc="Checking tile overlaps"):
        dtile = get_tile(mask_discard, y, x)
        dtile = skimage.morphology.remove_small_objects(dtile, max_size=1)
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
        logger.warn(
            f"Found {large_objects} large cells spanning an entire tile overlap"
            " that could not be segmented"
        )

    block_coords = [
        [(m,) + c for c in itertools.product(*(range(s) for s in labels_full.cdata_shape[1:]))]
        for m in (0, 1)
    ]
    outputs = [("cell", args.output_cell, block_coords[1])]
    if args.output_nucleus:
        outputs.append(("nucleus", args.output_nucleus, block_coords[0]))
    for name, out_path, coords in outputs:
        logger.info(f"Writing {name} masks to OME-TIFF: {out_path}")
        tifffile.imwrite(
            out_path,
            (pad_block(labels_full, c) for c in coords),
            shape=labels_full.shape[1:],
            dtype=labels_full.dtype,
            tile=(tw, tw),
            compression="zlib",
            predictor=True,
            maxworkers=1,
        )

    if args.output_discard:
        logger.info(f"Writing discard map to OME-TIFF: {args.output_discard}")
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


if __name__ == '__main__':
    main()
