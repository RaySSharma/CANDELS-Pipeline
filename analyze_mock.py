# code to do source detection, segmentation, and morphology measurements from simple mock images
# - Greg Snyder, Ray Sharma 2019

import astropy
import astropy.io.fits as fits
import astropy.table
import numpy as np
import photutils
from photutils.utils import calc_total_error
from photutils.segmentation import SegmentationImage
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel
import statmorph


# Run basic source detection
def detect_sources(input_name, input_ext_name, gain=2.4, photfnu=1.5e-07):
    with fits.open(input_name) as f:
        data = f[input_ext_name].data
        header = f[input_ext_name].header

    try:
        photfnu = header["PHOTFNU"]
    except KeyError:
        pass

    # Convert nJy to counts
    data /= 1e9 * photfnu * gain

    # build kernel for pre-filtering.  How big?
    # don't assume redshift knowledge here
    typical_kpc_per_arcsec = 8.0

    kernel_kpc_fwhm = 5.0
    kernel_arcsec_fwhm = kernel_kpc_fwhm / typical_kpc_per_arcsec
    kernel_pixel_fwhm = kernel_arcsec_fwhm / header["PIXSIZE"]

    sigma = kernel_pixel_fwhm * gaussian_fwhm_to_sigma
    nsize = int(5 * kernel_pixel_fwhm)
    kernel = Gaussian2DKernel(sigma, x_size=nsize, y_size=nsize)

    bkg_estimator = photutils.background.MedianBackground()
    bkg = photutils.Background2D(
        data, (25, 25), filter_size=(3, 3), bkg_estimator=bkg_estimator
    )
    thresh = bkg.background + (1.3 * bkg.background_rms)
    segmap_obj = photutils.detect_sources(
        data, thresh, npixels=16, filter_kernel=kernel
    )

    # No segmap found
    if segmap_obj is None:
        return None, None, None

    # Error image can be computed with photutils plus a GAIN keyword -- ratio of flux units to counts
    errmap = calc_total_error(data, bkg.background_rms, effective_gain=gain)

    segmap = segmap_obj.data
    props = photutils.source_properties(data, segmap, errmap)
    props_table = astropy.table.Table(props.to_table())
    # these give problems given their format/NoneType objects
    props_table.remove_columns(
        [
            "sky_centroid",
            "sky_centroid_icrs",
            "source_sum_err",
            "background_sum",
            "background_mean",
            "background_at_centroid",
        ]
    )

    output_hdu(input_name, "SEGMAP", segmap)
    output_hdu(input_name, "SEGMAP_PROPS", props_table, table=True)
    output_hdu(input_name, "WEIGHT_MAP", errmap)
    return segmap_obj, errmap, kernel, bkg


# Run PhotUtils Deblender
def deblend_sources(
    input_name, segm_obj, kernel, errmap, input_ext_name, gain=2.4, photfnu=1.5e-07
):
    with fits.open(input_name) as f:
        data = f[input_ext_name].data
        header = f[input_ext_name].header

    try:
        photfnu = header["PHOTFNU"]
    except KeyError:
        pass

    # Convert nJy to counts
    data /= 1e9 * photfnu * gain

    # No segmap found, cannot deblend
    if segm_obj is None:
        return None

    segm_obj = photutils.deblend_sources(
        data, segm_obj, npixels=16, filter_kernel=kernel
    )
    segmap = segm_obj.data

    props = photutils.source_properties(data, segmap, errmap)
    props_table = astropy.table.Table(props.to_table())
    # these give problems given their format/NoneType objects
    props_table.remove_columns(
        [
            "sky_centroid",
            "sky_centroid_icrs",
            "source_sum_err",
            "background_sum",
            "background_mean",
            "background_at_centroid",
        ]
    )

    output_hdu(input_name, "DEBLEND", segmap)
    output_hdu(input_name, "DEBLEND_PROPS", props_table, table=True)
    return segm_obj


def fill_bright_pixels(in_image, input_ext_name, sigma=1e4):
    from photutils.detection import find_peaks
    from astropy.stats import sigma_clipped_stats

    with fits.open(in_image) as fo:
        data = fo[input_ext_name].data
        header = fo[input_ext_name].header

    _, median, std = sigma_clipped_stats(data, sigma=sigma)
    threshold = median + (5 * std)
    tbl = find_peaks(data, threshold, box_size=11)

    xx = np.asarray(tbl["x_peak"])
    yy = np.asarray(tbl["y_peak"])
    for (x, y) in zip(*[xx, yy]):
        window = data[x - 1 : x, y - 1 : y]
        data[xx, yy] = np.median(window)
    output_hdu(in_image, "REALSIM_SMOOTH", data, header)


# Run morphology code
def source_morphology(
    in_image,
    input_ext_name,
    segm_obj=None,
    errmap=None,
    bkg=None,
    gain=2.4,
    photfnu=1.5e-07,
    **kwargs
):
    from scipy.stats import mode

    if segm_obj is None:
        with fits.open(in_image) as fo:
            try:
                segm_obj = SegmentationImage(fo["DEBLEND"].data)
                errmap = fo["WEIGHT_MAP"].data
                # seg_props = fo["DEBLEND_PROPS"].data
            except KeyError as err:
                print(err, "-", "Segmaps not in fits file")
                return None

    with fits.open(in_image) as fo:
        im = fo[input_ext_name].data
        header = fo[input_ext_name].header

    try:
        photfnu = header["PHOTFNU"]
    except KeyError:
        pass

    # Convert nJy to counts
    im /= 1e9 * photfnu * gain

    if bkg is None:
        import photutils

        bkg_estimator = photutils.background.MedianBackground()
        bkg = photutils.Background2D(
            im, (25, 25), filter_size=(3, 3), bkg_estimator=bkg_estimator
        )

    im -= bkg.background

    npix = im.shape[0]
    center_slice = segm_obj.data[
        int(npix / 2) - 3 : int(npix / 2) + 3, int(npix / 2) - 3 : int(npix / 2) + 3
    ]
    center_slice = center_slice[center_slice > 0].ravel()

    try:
        central_label = mode(center_slice).mode[0]
        # central_index = np.where(seg_props["id"] == center_slice[0, 0])[0][0]
        source_morph = statmorph.SourceMorphology(
            im, segm_obj, central_label, weightmap=errmap, **kwargs
        )
        return source_morph
    except (KeyError, IndexError, AttributeError, ValueError, TypeError) as err:
        print(err, "-", in_image, "not processed, skipping fit.")
        return None


def save_morph_params(in_image, source_morph, **kwargs):
    nhdu = fits.ImageHDU()

    if source_morph is not None:
        if kwargs is not None:
            for key, value in kwargs.items():
                nhdu.header[key] = source_morph[value]
    output_hdu(in_image, "SOURCE_MORPH", nhdu.data, nhdu.header)


def output_hdu(input_name, ext_name, data, header=None, table=False):
    hdu_extnames = np.asarray(fits.info(input_name, output=False), dtype=object).T[1]

    if header is None:
        header = fits.Header()

    header["EXTNAME"] = ext_name

    if ext_name in hdu_extnames:
        if table:
            fits.update(input_name, data.as_array(), header, extname=ext_name)
        else:
            fits.update(input_name, data, header, extname=ext_name)
    else:
        with fits.open(input_name, "append") as hdul:
            if table:
                hdu_out = fits.BinTableHDU(data, header=header)
            else:
                hdu_out = fits.ImageHDU(data, header=header)
            hdul.append(hdu_out)
