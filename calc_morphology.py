#!/usr/bin/env python3
import analyze_mock as am
import glob
import warnings
from astropy.io.fits.verify import VerifyWarning

warnings.simplefilter("ignore", category=VerifyWarning)

# List of morphological parameters to calculate in statmorph
morph_params = {
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
image_loc = "/scratch/rss230/AGN-Obscuration/outputs/*/*/WFC3_F160W/[3-5]/"
image_files = glob.glob(image_loc + "*.image.*.fits")

image_files.sort()

for i, image_mock in enumerate(image_files):
    print("Image Number:", i, "/", len(image_files), flush=True)
    print("Mock Image:", image_mock, flush=True)

    try:
        source_morph = am.source_morphology(
            image_mock, ext_name="RealSim",
        )  # Calculate morphological parameters using statmorph

        am.save_morph_params(
            image_mock, source_morph, **morph_params
        )  # Save morph params to HDU, generate statmorph image of params
    except (KeyError, IndexError, AttributeError, ValueError) as err:
        print(err, "-", image_mock, "not processed, skipping fit.")
        continue
