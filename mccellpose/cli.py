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


def segment_tile(timg, cp_model, contrast_limits, cytoplasm_thickness, diameter):
    if np.ptp(timg) == 0:
        return np.zeros(timg.shape, dtype="int32"), np.zeros(timg.shape, dtype="int32")

    timg = skimage.exposure.rescale_intensity(
        timg, in_range=contrast_limits, out_range="float"
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


def write_label_pyramid(level0, out_path, pixel_size, tile, predictor=True):
    """Write a 2-D label array as a tiled, pyramidal OME-TIFF with pixel size metadata"""
    x = level0 if isinstance(level0, dask.array.Array) else dask.array.from_zarr(level0)
    dtype = x.dtype
    base_shape = tuple(x.shape)
    # Number of factor-2 levels needed to bring the largest dimension down to a
    # single tile. Each coarser level's shape is the previous level's shape
    # halved and rounded up, matching the strided downsampling below.
    num_levels = max(int(np.ceil(np.log2(max(base_shape) / tile))) + 1, 1)
    shapes = [tuple(-(-s // 2 ** i) for s in base_shape) for i in range(num_levels)]

    def base_tiles():
        h, w = base_shape
        for r in range(0, h, tile):
            for c in range(0, w, tile):
                yield np.ascontiguousarray(x[r : r + tile, c : c + tile])

    def subres_tiles(level):
        # Build this level by reading the previous level back from the output
        # file as it is being written. is_ome=False because the OME-XML is only
        # finalised on writer close.
        tiff = tifffile.TiffFile(out_path, is_ome=False)
        try:
            prev = zarr.open(tiff.series[0].aszarr(level=level - 1), mode="r")
            h, w = prev.shape
            step = tile * 2
            for r in range(0, h, step):
                for c in range(0, w, step):
                    # Downsample by strided slicing rather than averaging so
                    # label IDs are preserved.
                    block = prev[r : r + step, c : c + step]
                    yield np.ascontiguousarray(block[::2, ::2])
        finally:
            tiff.close()

    opts = dict(
        tile=(tile, tile),
        # zstd is both faster and ~20% smaller than zlib.
        compression="zstd",
        resolution=(1e4 / pixel_size, 1e4 / pixel_size),
        resolutionunit="CENTIMETER",
        # Leave maxworkers at tifffile's default rather than forcing 1. It
        # compresses tiles across ~half the available cores -- CPU-affinity
        # aware on Linux, None-safe, and overridable via TIFFFILE_NUM_THREADS.
        # Pyramid writing is its own phase after segmentation, so the cores are
        # otherwise idle, and ~half-cores already sits near the speedup knee.
    )
    if predictor:
        opts["predictor"] = True
    with tifffile.TiffWriter(out_path, bigtiff=True, ome=True) as tif:
        tif.write(
            base_tiles(),
            shape=base_shape,
            dtype=dtype,
            subifds=num_levels - 1,
            metadata={
                "axes": "YX",
                "PhysicalSizeX": pixel_size,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_size,
                "PhysicalSizeYUnit": "µm",
            },
            **opts,
        )
        for level in range(1, num_levels):
            tif.write(
                subres_tiles(level),
                shape=shapes[level],
                dtype=dtype,
                subfiletype=1,
                **opts,
            )


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
        '--contrast-limits',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        help="Intensity value limits for pre-scaling the image before"
        " segmentation. Auto-detected from the image if not specified.",
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
        logger.info(f"Pixel size detected from OME-TIFF: {pixel_size} µm")

    tw = args.tile_width
    if tw % 16 != 0:
        logger.error("--tile-width value must be a multiple of 16")
        sys.exit(1)

    overlap = round(args.tile_overlap / pixel_size)
    logger.info(f"Tile overlap: {args.tile_overlap} µm ({overlap} px)")
    if overlap < 3:
        logger.warn(
            "Tile overlap is very small (less than 3 pixels) -- many cells are"
            " likely to be missed"
        )
    diameter = args.diameter / pixel_size
    logger.info(f"Expected nucleus diameter: {args.diameter} µm ({diameter} px)")

    img = zarr.open(tiff.series[0][args.channel - 1].aszarr(level=0), mode="r")
    expand_size_px = round(args.expand_size / pixel_size)

    if args.contrast_limits:
        contrast_limits = tuple(args.contrast_limits)
        logger.info(f"Rescaling intensity to user-specified limits: {contrast_limits}")
    else:
        logger.info("Computing image contrast...")
        intensity_max = float(auto_threshold(dask.array.from_zarr(img)))
        contrast_limits = (0, intensity_max)
        logger.info(f"Rescaling intensity to auto-detected limits: {contrast_limits}")

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
            get_tile(img, y, x), cp_model, contrast_limits, expand_size_px, diameter
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

    # channel 0 = nucleus, 1 = cell in the labels_full zarr
    outputs = [("cell", args.output_cell, 1)]
    if args.output_nucleus:
        outputs.append(("nucleus", args.output_nucleus, 0))
    labels_da = dask.array.from_zarr(labels_full)
    for name, out_path, m in outputs:
        logger.info(f"Writing {name} masks to pyramidal OME-TIFF: {out_path}")
        write_label_pyramid(
            labels_da[m],
            out_path,
            pixel_size,
            tw,
            predictor=True,
        )

    if args.output_discard:
        logger.info(f"Writing discard map to pyramidal OME-TIFF: {args.output_discard}")
        write_label_pyramid(
            dask.array.from_zarr(mask_discard),
            args.output_discard,
            pixel_size,
            tw,
            predictor=False,
        )


if __name__ == '__main__':
    main()
