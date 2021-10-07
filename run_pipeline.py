#!/usr/bin/env python3
from os.path import isfile
from sys import argv

import matplotlib
import numpy as np

import analyze_mock as am
import pristine_to_mock as ptm
import RealSim.ObsRealism as obs

matplotlib.use("Agg")

# {filter: central wavelength [micron], instrument resolution [arcsec], gain [e-/DN]}
FILT_WHEEL = {
    "WFC3_F160W": [1.5369, 0.18, 2.4],
}
# CANDELS RealSim parameters [catalog, field, mean exp time(s), photfnu]
FIELD_INFO = {
    "UDS": [
        "hlsp_candels_hst_wfc3_uds-tot-multiband_f160w_v1_cat.fits",
        "hlsp_candels_hst_wfc3_uds-tot_f160w_v1.0_drz.fits",
        3300,
        1.5053973e-07,
    ],
    "EGS": [
        "hlsp_candels_hst_wfc3_egs-tot-multiband_f160w_v1_cat.fits",
        "hlsp_candels_hst_wfc3_egs-tot-60mas_f160w_v1.0_drz.fits",
        3200,
        1.518757e-07,
    ],
    "GS": [
        "hlsp_candels_hst_wfc3_goodss-tot-multiband_f160w_v1_cat.fits",
        "hlsp_candels_hst_wfc3_gs-tot_f160w_v1.0_drz.fits",
        3550,
        1.5053973e-07,
    ],
    "COS": [
        "hlsp_candels_hst_wfc3_cos-tot-multiband_f160w_v1_cat.fits",
        "hlsp_candels_hst_wfc3_cos-tot_f160w_v1.0_drz.fits",
        3200,
        1.5053973e-07,
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
REALSIM_INPUT_DIR = "/scratch/rss230/CANDELS-Pipeline/RealSim/Inputs/"
RANDOMIZE_CANDELS_FIELD = False
GENERATE_MOCK = False
GENERATE_REALSIM = False
GENERATE_SEG = True
DEBLEND = True
GENERATE_MORPH = True
MASK_AGN = True


def choose_candels_field():
    if RANDOMIZE_CANDELS_FIELD:
        field_name = np.random.choice(list(FIELD_INFO.keys()))
    else:
        try:
            import astropy.io.fits as fits

            field_name = fits.getheader(image, "RealSim")["CANDELS_FIELD"]
        except:
            field_name = np.random.choice(list(FIELD_INFO.keys()))
    return field_name


if __name__ == "__main__":
    image = argv[1]
    if not isfile(image):
        raise (FileNotFoundError)

    print("Image:", image, flush=True)

    deblend_seg, deblend_seg_props, errmap = None, None, None
    candels_field = choose_candels_field()

    if GENERATE_MOCK:
        ptm.convolve_with_fwhm(image, FILT_WHEEL)  # Convolve mock image with PSF
        ptm.add_shot_noise(
            image, field_info=FIELD_INFO, field_name=candels_field
        )  # Add noise model

    if GENERATE_REALSIM:
        candels_args = obs.make_candels_args(
            field_info=FIELD_INFO, input_dir=REALSIM_INPUT_DIR, field_name=candels_field
        )
        obs.ObsRealism(image, candels_args)  # Apply RealSim CANDELS fields

    gain = FILT_WHEEL["WFC3_F160W"][2]

    if GENERATE_SEG:
        seg, errmap, kernel, sky_background = am.detect_sources(
            image, input_ext_name="RealSim", gain=gain
        )  # Run source detection with photutils

    if GENERATE_MORPH:
        if DEBLEND:
            seg = am.deblend_sources(
                image,
                segm_obj=seg,
                kernel=kernel,
                errmap=errmap,
                input_ext_name="RealSim",
                gain=gain,
            )  # Deblend detected sources

        if MASK_AGN:
            mask = am.mask_agn(image, "RealSim", sigma=1e4)
        else:
            mask = None

        source_morph = am.source_morphology(
            image,
            input_ext_name="RealSim",
            segm_obj=seg,
            errmap=errmap,
            bkg=sky_background,
            gain=gain,
            mask=mask,
            sersic_maxiter=1000
        )  # Calculate morphological parameters using statmorph with chosen segmap

        am.save_morph_params(
            image, source_morph, **MORPH_PARAMS
        )  # Save morph params to HDU, generate statmorph image of params
