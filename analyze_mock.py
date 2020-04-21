# code to do source detection, segmentation, and morphology measurements from simple mock images
# - Greg Snyder, Ray Sharma 2019

import astropy
import astropy.io.fits as fits
import astropy.table
import numpy as np
import photutils
from photutils.utils import calc_total_error
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel
import statmorph
from statmorph.utils.image_diagnostics import make_figure


# Run basic source detection
def detect_sources(in_image, ext_name, filt_wheel, **kwargs):
    fo = fits.open(in_image, 'append')
    hdu = fo[ext_name]

    # build kernel for pre-filtering.  How big?
    # don't assume redshift knowledge here
    typical_kpc_per_arcsec = 8.0

    kernel_kpc_fwhm = 5.0
    kernel_arcsec_fwhm = kernel_kpc_fwhm / typical_kpc_per_arcsec
    kernel_pixel_fwhm = kernel_arcsec_fwhm / hdu.header['PIXSIZE']

    sigma = kernel_pixel_fwhm * gaussian_fwhm_to_sigma
    nsize = int(5 * kernel_pixel_fwhm)
    kernel = Gaussian2DKernel(sigma, x_size=nsize, y_size=nsize)

    bkg_estimator = photutils.MedianBackground()
    bkg = photutils.Background2D(hdu.data, (50, 50),
                                 bkg_estimator=bkg_estimator)
    thresh = bkg.background + (5. * bkg.background_rms)
    segmap_obj = photutils.detect_sources(hdu.data, thresh, npixels=10,
                                          filter_kernel=kernel, **kwargs)

    if segmap_obj is None:
        nhdu = fits.ImageHDU()
        nhdu.header['EXTNAME'] = 'SEGMAP'
        fo.append(nhdu)

        thdu = fits.BinTableHDU()
        thdu.header['EXTNAME'] = 'SEGMAP_PROPS'
        fo.append(thdu)

        fo.flush()
        fo.close()
        return None, None, None

    # Error image can be computed with photutils plus a GAIN keyword -- ratio of flux units to counts
    gain = filt_wheel[hdu.header['FILTER']][2]
    errmap = calc_total_error(hdu.data, bkg.background_rms, effective_gain=gain)

    segmap = segmap_obj.data
    props = photutils.source_properties(hdu.data, segmap, errmap)
    props_table = astropy.table.Table(props.to_table())
    # these give problems given their format/NoneType objects
    props_table.remove_columns([
        'sky_centroid', 'sky_centroid_icrs', 'source_sum_err',
        'background_sum', 'background_mean', 'background_at_centroid'
    ])
    nhdu = fits.ImageHDU(segmap)

    # save segmap and info
    nhdu.header['EXTNAME'] = 'SEGMAP'
    fo.append(nhdu)

    thdu = fits.BinTableHDU(props_table)
    thdu.header['EXTNAME'] = 'SEGMAP_PROPS'
    fo.append(thdu)

    fo.flush()
    fo.close()
    return segmap_obj, kernel, errmap


# Run PhotUtils Deblender
def deblend_sources(in_image, segm_obj, kernel, errmap, ext_name):
    fo = fits.open(in_image, 'append')
    hdu = fo[ext_name]

    if segm_obj is None:
        nhdu = fits.ImageHDU()

        # save segmap and info
        nhdu.header['EXTNAME'] = 'DEBLEND'

        thdu = fits.BinTableHDU()
        thdu.header['EXTNAME'] = 'DEBLEND_PROPS'

        fo.append(nhdu)
        fo.append(thdu)

        fo.flush()
        fo.close()
        return None

    segm_obj = photutils.deblend_sources(hdu.data, segm_obj, npixels=10,
                                         filter_kernel=kernel, nlevels=32,
                                         contrast=0.01)
    segmap = segm_obj.data

    props = photutils.source_properties(hdu.data, segmap, errmap)
    props_table = astropy.table.Table(props.to_table())
    # these give problems given their format/NoneType objects
    props_table.remove_columns([
        'sky_centroid', 'sky_centroid_icrs', 'source_sum_err',
        'background_sum', 'background_mean', 'background_at_centroid'
    ])
    nhdu = fits.ImageHDU(segmap)

    # save segmap and info
    nhdu.header['EXTNAME'] = 'DEBLEND'

    thdu = fits.BinTableHDU(props_table)
    thdu.header['EXTNAME'] = 'DEBLEND_PROPS'

    fo.append(nhdu)
    fo.append(thdu)

    fo.flush()
    fo.close()
    return segm_obj


# Run morphology code
def source_morphology(in_image, segm_obj, errmap, ext_name,
                      props_ext_name):
    fo = fits.open(in_image, 'append')
    hdu = fo[ext_name]
    seg_props = fo[props_ext_name].data

    if segm_obj is None:
        return None

    npix = hdu.data.shape[0]
    center_slice = segm_obj.data[int(npix / 2) - 2:int(npix / 2) + 2,
                                 int(npix / 2) - 2:int(npix / 2) + 2]

    segmap = segm_obj.data
    central_index = seg_props['id'] == center_slice[0, 0]

    source_morph = statmorph.source_morphology(hdu.data, segmap, weightmap=errmap)
    fo.close()
    try:
        return np.array(source_morph)[central_index][0]
    except IndexError:
        return None


def save_morph_params(in_image, source_morph, fig_name, **kwargs):
    fo = fits.open(in_image, 'append')
    nhdu = fits.ImageHDU()
    nhdu.header['EXTNAME'] = 'SOURCE_MORPH'

    if source_morph is not None:
        nhdu.data = source_morph._segmap_gini.astype(int)
        if kwargs is not None:
            for key, value in kwargs.items():
                nhdu.header[key] = source_morph[value]
        fo.append(nhdu)
        fo.flush()
        fo.close()
        fig = make_figure(source_morph)
        fig.savefig(fig_name, dpi=150)
    else:
        fo.append(nhdu)
        fo.flush()
        fo.close()
