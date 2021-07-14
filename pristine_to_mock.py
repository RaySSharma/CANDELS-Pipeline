# code for converting pristine simulated images to realistic mocks
# - Greg Snyder, Ray Sharma 2019

import numpy as np
import scipy as sp
import astropy.cosmology
import astropy.io.fits as fits
from astropy.stats import gaussian_fwhm_to_sigma
from analyze_mock import output_hdu

# Convolve image with diffraction-limited PSF FWHM
def convolve_with_fwhm(in_image, filt_wheel):
    with fits.open(in_image) as f:
        image_in = f["SimulatedImage"].data
        header_in = f["SimulatedImage"].header

    # Select filter wheel
    filt = filt_wheel[header_in["FILTER"]]

    # Set diffraction-limited FWHM, in arcseconds
    fwhm_arcsec = filt[1]

    # Cosmology used by Choi et al. 2018
    redshift = header_in["REDSHIFT"]
    cosmology = astropy.cosmology.FlatLambdaCDM(0.72 * 100.0, 0.26, Ob0=0.044)
    kpc_per_arcsec = cosmology.kpc_proper_per_arcmin(redshift).value / 60.0

    # Redshift dim simulated image
    # A little tricky, input units here are nJy [~flux/Hz]
    # If input units were [~flux/Angstrom], dimming would be (1+z)^5
    z_dim = (1 + redshift) ** 3
    image_in /= z_dim

    # calculate PSF width in pixel units
    pixel_size_arcsec = 2 * header_in["PIX_KPC"] / kpc_per_arcsec
    sigma_arcsec = fwhm_arcsec * gaussian_fwhm_to_sigma
    sigma_pixels = sigma_arcsec / pixel_size_arcsec

    image_out = sp.ndimage.filters.gaussian_filter(
        image_in, sigma_pixels, mode="nearest"
    )

    header_out = header_in.copy()
    header_out["FWHMPIX"] = (sigma_pixels / gaussian_fwhm_to_sigma, "pixels")
    header_out["SIGMAPIX"] = (sigma_pixels, "pixels")
    header_out["FWHM"] = (fwhm_arcsec, "arcsec")
    header_out["SIGMA"] = (sigma_arcsec, "arcsec")
    header_out["PIXSIZE"] = (pixel_size_arcsec, "arcsec")
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

    header_out = header_in.copy()
    header_out["SBLIM"] = (sb_maglim, "mag/arcsec^2")
    header_out["RMSNOISE"] = (sigma_njy, "nanojanskies")
    output_hdu(in_image, "MockImage_sky", data=image_out, header=header_out)


# Converts nJy to e using CANDELS exptime + inverse sensitivity "PHOTFNU", calculates source shot noise, then converts back
def add_shot_noise(in_image, field_info, field_name):
    image_in, header_in= fits.getdata(in_image, "MockImage_Noiseless", header=True)
    exp = field_info[field_name][2]
    photfnu = field_info[field_name][3]

    calibration = exp / 1e9 / photfnu
    image_out = (
        np.random.poisson(lam=image_in * calibration) / calibration
    )  # Calculate poisson noise in electrons
    header_out = header_in.copy()
    header_out["field"] = field_name
    header_out["exptime"] = exp
    header_out["photfnu"] = photfnu
    output_hdu(in_image, "MockImage", data=image_out, header=header_out)