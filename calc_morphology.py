#!/usr/bin/env python3
import analyze_mock as am
import glob

# {filter: central wavelength [micron], instrument resolution [arcsec], gain}
filt_wheel = {
    "WFC3_F160W": [1.5369, 0.18, 2.4],
}
# CANDELS RealSim catalog/field parameters
field_info = {
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
realsim_input_dir = "/scratch/rss230/CANDELS-Pipeline/RealSim/Inputs/"

# mag/arcsec^2
detection_limits = [25]

# List of morphological parameters to calculate in statmorph
morph_params = {
    "GINI": "gini",
    "M20": "m20",
    "CONC": "concentration",
    "ASYM": "asymmetry",
    "SMOOTH": "smoothness",
    "SERSIC_N": "sersic_n",
    "M": "multimode",
    "D": "deviation",
    "I": "intensity",
    "FLAG": "flag",
    "FLAG_SERSIC": "flag_sersic",
}
image_loc = "/scratch/rss230/AGN-Obscuration/outputs/*/*/WFC3_F160W/[3-5]/"
image_files = glob.glob(image_loc + "*.image.fits")

for i, image in enumerate(image_files):
    print("Image Number:", i, "/", len(image_files), flush=True)
    print("Image:", image, flush=True)
    for lim in detection_limits:
        image_mock = image[:-5] + ".SB" + str(lim) + ".fits"
        fig_name = image[:-5] + ".SB" + str(lim) + ".png"

        print("SB:", lim, "mag arcsec^-2", flush=True)
        print("Mock Image:", image_mock, flush=True)

        try:
            source_morph = am.source_morphology(
                image_mock,
                ext_name="RealSim",
            )  # Calculate morphological parameters using statmorph

            am.save_morph_params(
                image_mock, source_morph, fig_name=fig_name, **morph_params
            )  # Save morph params to HDU, generate statmorph image of params
        except KeyError as err:
            print(err, '-', image_mock, 'not processed, skipping fit.')
            continue