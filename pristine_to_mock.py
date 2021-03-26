# code for converting pristine simulated images to realistic mocks
# - Greg Snyder, Ray Sharma 2019

import numpy as np
import scipy.ndimage
import scipy as sp
import astropy.cosmology
import astropy.constants as constants
import astropy.io.fits as fits
import astropy.units as u
from astropy.stats import gaussian_fwhm_to_sigma

'''
def output_pristine_fits_image(in_image, filt_wheel):
    # Input image in units lambda*F_lambda [erg/s/cm^2]
    im = fits.getdata(in_image, "SimulatedImage")
    header = fits.getheader(in_image, "SimulatedImage")

    # Select filter wheel
    filt = filt_wheel[header["FILTER"]]
    lambda_eff = filt[0]

    lambda_eff_microns = lambda_eff * u.micron
    nu_eff_hz = constants.c.to(u.micron / u.s) / lambda_eff_microns

    # Convert to F_nu = lambda*F_lambda/nu
    image_flux = (im * u.erg / (u.s * u.cm * u.cm)) / (nu_eff_hz)

    # Convert to NanoJanskies
    image_njy = image_flux.to("nJy")

    # Redshift dim image by (1+z)**5
    z_dim = (1 + header["redshift"])**5
    image_njy /= z_dim

    output_hdu(in_image, "SimulatedImage", data=image_njy, header=header)

'''
# Convolve image with diffraction-limited PSF FWHM
def convolve_with_fwhm(in_image, filt_wheel):
    # load image data and metadata
    image_in = fits.getdata(in_image, "SimulatedImage")
    header_in = fits.getheader(in_image, "SimulatedImage")

    # Select filter wheel
    filt = filt_wheel[header_in["FILTER"]]

    # Set diffraction-limited FWHM, in arcseconds
    fwhm_arcsec = filt[1]

    # Cosmology used by Choi et al. 2018
    redshift = header_in["REDSHIFT"]
    cosmology = astropy.cosmology.FlatLambdaCDM(0.72 * 100.0, 0.26, Ob0=0.044)
    kpc_per_arcsec = cosmology.kpc_proper_per_arcmin(redshift).value / 60.0

    # Redshift dim simulated image
    z_dim = (1 + redshift)**5
    image_in /= z_dim

    # calculate PSF width in pixel units
    pixel_size_arcsec = 2 * header_in["PIX_KPC"] / kpc_per_arcsec
    sigma_arcsec = fwhm_arcsec * gaussian_fwhm_to_sigma
    sigma_pixels = sigma_arcsec / pixel_size_arcsec

    image_out = sp.ndimage.filters.gaussian_filter(
        image_in, sigma_pixels, mode="nearest"
    )

    header_out = header_in
    header_in["FWHMPIX"] = (sigma_pixels / gaussian_fwhm_to_sigma, "pixels")
    header_in["SIGMAPIX"] = (sigma_pixels, "pixels")
    header_in["FWHM"] = (fwhm_arcsec, "arcsec")
    header_in["SIGMA"] = (sigma_arcsec, "arcsec")
    header_in["PIXSIZE"] = (pixel_size_arcsec, "arcsec")

    output_hdu(in_image, "MockImage_Noiseless", data=image_out, header=header_out)


# accepts sb_maglim value which corresponds to magnitudes per square arcsecond
def add_simple_noise(in_image, sb_maglim, alg="Snyder2019"):
    image_in = fits.getdata(in_image, "MockImage_Noiseless")
    header_in = fits.getheader(in_image, "MockImage_Noiseless")

    if (
        alg == "Snyder2019"
    ):  # algorithm from Snyder et al. (2019), SB_limit \sim 5sigma above the background
        sigma_njy = (
            (2.0 ** (-0.5))
            * ((1.0e9) * (3631.0 / 5.0) * 10.0 ** (-0.4 * sb_maglim))
            * header_in["PIXSIZE"]
            * (3.0 * header_in["FWHM"])
        )
    else:
        sigma_njy = 0

    npix = image_in.shape[0]

    noise_image = sigma_njy * np.random.randn(npix, npix)

    image_out = image_in + noise_image

    header_out = header_in
    header_out["SBLIM"] = (sb_maglim, "mag/arcsec^2")
    header_out["RMSNOISE"] = (sigma_njy, "nanojanskies")
    
    output_hdu(in_image, "MockImage", data=image_out, header=header_out)


def output_hdu(input_name, ext_name, data, header=None, table=False):
    hdu_extnames = np.asarray(fits.info(input_name, output=False)).T[1]
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
            hdu_out.header["EXTNAME"] = ext_name
            hdul.append(hdu_out)

