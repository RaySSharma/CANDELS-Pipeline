# code for converting pristine simulated images to realistic mocks
import glob
import sys
import os
import photutils

import astropy.cosmology
import astropy.constants as constants
import astropy.io.fits as fits
import astropy.units as u
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np
import scipy.ndimage
import scipy as sp

# {filter: central wavelength [micron], instrument resolution [arcsec]}
filt_wheel = {
    'ACS_F814W': [0.8353, 0.05],
    'WFC3_F160W': [1.5369, 0.13],
    'ACS_F606W': [0.5907, 0.05]
}

image_files = glob.glob('*.image.fits')


def output_pristine_fits_image(image_file, out_file):
    # Input image in units lambda*F_lambda [erg/s/cm^2]
    fo = fits.open(image_file, 'readonly')

    outfo = fits.HDUList()
    outfo.append(fo[0])

    im = fo[0].data
    header = fo[0].header

    # Select filter wheel
    filt = filt_wheel[fo[0].header['FILTER']]
    lambda_eff = filt[0]
    fwhm_arcsec = filt[1]

    lambda_eff_microns = lambda_eff * u.micron
    nu_eff_hz = constants.c.to(u.micron / u.s) / lambda_eff_microns

    # Convert to F_nu = lambda*F_lambda/nu
    image_flux = (im * u.erg / (u.s * u.cm * u.cm)) / (nu_eff_hz)

    # Convert to NanoJanskies
    image_njy = image_flux.to('nJy')

    image_hdu = fits.ImageHDU(image_njy.value, header=header)

    image_hdu.header['EXTNAME'] = 'SimulatedImage'
    image_hdu.header['BUNIT'] = 'nanojanskies'

    # Set diffraction-limited FWHM, in arcseconds
    image_hdu.header['FWHM'] = (fwhm_arcsec, 'arcsec')

    outfo.append(image_hdu)
    outfo.writeto(out_file, overwrite=True)

    return


# Convolve image with diffraction-limited PSF FWHM
def convolve_with_fwhm(in_image):
    # open pristine image fits file
    in_fo = fits.open(in_image, 'append')

    # load image data and metadata
    image_in = in_fo['SimulatedImage'].data
    header_in = in_fo['SimulatedImage'].header
    fwhm_arcsec = in_fo['SimulatedImage'].header['FWHM']

    # Cosmology used by Choi et al. 2018
    cosmology = astropy.cosmology.FlatLambdaCDM(0.72 * 100.0, 0.26, Ob0=0.044)
    kpc_per_arcsec = cosmology.kpc_proper_per_arcmin(
        header_in['REDSHIFT']).value / 60.0

    # calculate PSF width in pixel units
    pixel_size_arcsec = header_in['PIX_KPC'] / kpc_per_arcsec
    sigma_arcsec = fwhm_arcsec * gaussian_fwhm_to_sigma
    sigma_pixels = sigma_arcsec / pixel_size_arcsec

    image_out = sp.ndimage.filters.gaussian_filter(image_in, sigma_pixels,
                                                   mode='nearest')
    hdu_out = fits.ImageHDU(image_out, header=header_in)
    hdu_out.header['FWHMPIX'] = (sigma_pixels / gaussian_fwhm_to_sigma,
                                 'pixels')
    hdu_out.header['SIGMAPIX'] = (sigma_pixels, 'pixels')
    hdu_out.header['SIGMA'] = (sigma_arcsec, 'arcsec')
    hdu_out.header['EXTNAME'] = 'MockImage_Noiseless'
    hdu_out.header['PIXSIZE'] = (pixel_size_arcsec, 'arcsec')

    in_fo.append(hdu_out)
    in_fo.flush()

    return


# accepts sb_maglim value which corresponds to magnitudes per square arcsecond
def add_simple_noise(in_image, sb_maglim=25.0, sb_label='25'):

    in_fo = fits.open(in_image, 'append')

    image_in = in_fo['MockImage_Noiseless'].data
    header_in = in_fo['MockImage_Noiseless'].header

    # algorithm from Snyder et al. (2019)
    sigma_njy = (2.0**(-0.5)) * ( (1.0e9) * (3631.0 / 5.0) * 10.0**(-0.4 * sb_maglim) ) * header_in['PIXSIZE'] * (3.0 * header_in['FWHM'])

    npix = image_in.shape[0]

    noise_image = sigma_njy * np.random.randn(npix, npix)

    image_out = image_in + noise_image

    hdu_out = fits.ImageHDU(image_out, header=header_in)
    hdu_out.header['EXTNAME'] = 'MockImage_SB' + sb_label
    hdu_out.header['SBLIM'] = (sb_maglim, 'mag/arcsec^2')
    hdu_out.header['RMSNOISE'] = (sigma_njy, 'nanojanskies')

    in_fo.append(hdu_out)
    in_fo.flush()

    return
