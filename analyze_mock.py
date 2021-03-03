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
def detect_sources(in_image, ext_name, filt_wheel, **kwargs):
    fo = fits.open(in_image, "append")
    hdu = fo[ext_name]

    # build kernel for pre-filtering.  How big?
    # don't assume redshift knowledge here
    typical_kpc_per_arcsec = 8.0

    kernel_kpc_fwhm = 5.0
    kernel_arcsec_fwhm = kernel_kpc_fwhm / typical_kpc_per_arcsec
    kernel_pixel_fwhm = kernel_arcsec_fwhm / hdu.header["PIXSIZE"]

    sigma = kernel_pixel_fwhm * gaussian_fwhm_to_sigma
    nsize = int(5 * kernel_pixel_fwhm)
    kernel = Gaussian2DKernel(sigma, x_size=nsize, y_size=nsize)

    bkg_estimator = photutils.MedianBackground()
    bkg = photutils.Background2D(hdu.data, (50, 50), bkg_estimator=bkg_estimator)
    thresh = bkg.background + (5.0 * bkg.background_rms)
    segmap_obj = photutils.detect_sources(
        hdu.data, thresh, npixels=5, filter_kernel=kernel, **kwargs
    )

    if segmap_obj is None:
        nhdu = fits.ImageHDU()
        nhdu.header["EXTNAME"] = "SEGMAP"
        fo.append(nhdu)

        thdu = fits.BinTableHDU()
        thdu.header["EXTNAME"] = "SEGMAP_PROPS"
        fo.append(thdu)

        fo.flush()
        fo.close()
        return None, None, None
    else:
        # Error image can be computed with photutils plus a GAIN keyword -- ratio of flux units to counts
        gain = filt_wheel[hdu.header["FILTER"]][2]
        errmap = calc_total_error(hdu.data, bkg.background_rms, effective_gain=gain)

        segmap = segmap_obj.data
        props = photutils.source_properties(hdu.data, segmap, errmap)
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
        nhdu = fits.ImageHDU(segmap)

        # save segmap and info
        nhdu.header["EXTNAME"] = "SEGMAP"
        fo.append(nhdu)

        thdu = fits.BinTableHDU(props_table)
        thdu.header["EXTNAME"] = "SEGMAP_PROPS"
        fo.append(thdu)

        fo.flush()

        nhdu = fits.ImageHDU(errmap)

        # save errmap
        nhdu.header["EXTNAME"] = "WEIGHT_MAP"
        fo.append(nhdu)

        fo.flush()
        fo.close()
    return segmap_obj, kernel, errmap


# Run PhotUtils Deblender
def deblend_sources(in_image, segm_obj, kernel, errmap, ext_name):
    fo = fits.open(in_image, "append")
    hdu = fo[ext_name]

    if segm_obj is None:
        nhdu = fits.ImageHDU()

        # save segmap and info
        nhdu.header["EXTNAME"] = "DEBLEND"

        thdu = fits.BinTableHDU()
        thdu.header["EXTNAME"] = "DEBLEND_PROPS"

        fo.append(nhdu)
        fo.append(thdu)

        fo.flush()
        fo.close()
        return None

    segm_obj = photutils.deblend_sources(
        hdu.data, segm_obj, npixels=5, filter_kernel=kernel
    )
    segmap = segm_obj.data

    props = photutils.source_properties(hdu.data, segmap, errmap)
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
    nhdu = fits.ImageHDU(segmap)

    # save segmap and info
    nhdu.header["EXTNAME"] = "DEBLEND"

    thdu = fits.BinTableHDU(props_table)
    thdu.header["EXTNAME"] = "DEBLEND_PROPS"

    fo.append(nhdu)
    fo.append(thdu)

    fo.flush()
    fo.close()
    return segm_obj


# Run morphology code
def source_morphology(in_image, ext_name, **kwargs):
    fo = fits.open(in_image, "append")
    hdu = fo[ext_name]

    segm_obj = SegmentationImage(fo["DEBLEND"].data)
    errmap = fo["WEIGHT_MAP"].data
    seg_props = fo["DEBLEND_PROPS"].data
    im = hdu.data

    bkg_estimator = photutils.MedianBackground()
    bkg = photutils.Background2D(hdu.data, (50, 50), bkg_estimator=bkg_estimator)
    im -= bkg.background

    npix = im.shape[0]
    center_slice = segm_obj.data[
        int(npix / 2) - 2 : int(npix / 2) + 2, int(npix / 2) - 2 : int(npix / 2) + 2
    ]

    central_index = np.where(seg_props["id"] == center_slice[0, 0])[0][0]

    fo.flush()
    fo.close()
    source_morph = statmorph.SourceMorphology(
        im, segm_obj, central_index, weightmap=errmap, **kwargs
    )
    return source_morph


def save_morph_params(in_image, source_morph, **kwargs):
    fo = fits.open(in_image, "append")
    nhdu = fits.ImageHDU()
    nhdu.header["EXTNAME"] = "SOURCE_MORPH"

    if source_morph is not None:
        if kwargs is not None:
            for key, value in kwargs.items():
                nhdu.header[key] = source_morph[value]
    fo.append(nhdu)
    fo.flush()
    fo.close()
