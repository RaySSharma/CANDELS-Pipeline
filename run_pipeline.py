#!/usr/bin/env python3
import pristine_to_mock as ptm
import analyze_mock as am
import RealSim.ObsRealism as obs
import glob
import matplotlib

matplotlib.use("Agg")

# {filter: central wavelength [micron], instrument resolution [arcsec], gain}
FILT_WHEEL = {
    "WFC3_F160W": [1.5369, 0.18, 2.4],
}
# CANDELS RealSim catalog/field parameters
FIELD_INFO = {
    "UDS": [
        "hlsp_candels_hst_wfc3_uds-tot-multiband_f160w_v1_cat.fits",
        "hlsp_candels_hst_wfc3_uds-tot_f160w_v1.0_drz.fits",
    ],
    "EGS": [
        "hlsp_candels_hst_wfc3_egs-tot-multiband_f160w_v1_cat.fits",
        "hlsp_candels_hst_wfc3_egs-tot-60mas_f160w_v1.0_drz.fits",
    ],
    "GS": [
        "hlsp_candels_hst_wfc3_goodss-tot-multiband_f160w_v1_cat.fits",
        "hlsp_candels_hst_wfc3_gs-tot_f160w_v1.0_drz.fits",
    ],
    "COS": [
        "hlsp_candels_hst_wfc3_cos-tot-multiband_f160w_v1_cat.fits",
        "hlsp_candels_hst_wfc3_cos-tot_f160w_v1.0_drz.fits",
    ],
}
# List of morphological parameters to calculate in statmorph
MORPH_PARAMS = {
    "GINI": "gini",
    "M20": "m20",
    "CONC": "concentration",
    "ASYM": "asymmetry",
    "SMOOTH": "smoothness",
    "SERSIC_N": "sersic_n",
    "SERSIC_A": "sersic_amplitude",
    "SERSIC_REFF": "sersic_rhalf",
    "SERSIC_XC": "sersic_xc",
    "SERSIC_YC": "sersic_yc",
    "SERSIC_ELLIP": "sersic_ellip",
    "SERSIC_THETA": "sersic_theta",
    "M": "multimode",
    "D": "deviation",
    "I": "intensity",
    "FLAG": "flag",
    "FLAG_SERSIC": "flag_sersic",
}
REALSIM_INPUT_DIR = "/data2/ramonsharma/RealSim/Inputs/"
IMAGE_DIR = "/data2/ramonsharma/backups/realsim_images/"
GENERATE_MOCK = True
GENERATE_REALSIM = True
GENERATE_SEG = True
GENERATE_MORPH = True

image_files = glob.glob(IMAGE_DIR + "*.fits")
deblend_seg, deblend_seg_props, errmap = None, None, None
for i, image in enumerate(image_files):
    print("Image Number:", i, "/", len(image_files), flush=True)
    print("Image:", image, flush=True)

    if GENERATE_MOCK:
        ptm.output_pristine_fits_image(
            image, image, FILT_WHEEL
        )  # Setup image for mock creation
        ptm.convolve_with_fwhm(image, FILT_WHEEL)  # Convolve mock image with PSF
        ptm.add_simple_noise(image, sb_maglim=25, alg="Snyder2019")  # Add noise model

    if GENERATE_REALSIM:
        candels_args = obs.make_candels_args(FIELD_INFO, REALSIM_INPUT_DIR)
        obs.ObsRealism(image, candels_args)  # Apply RealSim CANDELS fields

    if GENERATE_SEG:
        seg, kernel, errmap = am.detect_sources(
            image, input_ext_name="RealSim", filt_wheel=FILT_WHEEL
        )  # Run source detection with photutils

        deblend_seg, deblend_seg_props = am.deblend_sources(
            image, seg, kernel, errmap, input_ext_name="RealSim"
        )  # Deblend detected sources

    if GENERATE_MORPH:
        source_morph = am.source_morphology(
            image, "RealSim", deblend_seg, deblend_seg_props, errmap
        )  # Calculate morphological parameters using statmorph

        am.save_morph_params(
            image, source_morph, **MORPH_PARAMS
        )  # Save morph params to HDU, generate statmorph image of params
        
