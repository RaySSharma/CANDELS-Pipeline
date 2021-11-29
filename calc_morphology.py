#!/usr/bin/env python3
import analyze_mock as am
import numpy as np
import pathlib
import warnings
import pandas as pd
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
    "SN": "sn_per_pixel",
    "FLAG": "flag",
    "FLAG_SERSIC": "flag_sersic",
}
image_loc = "/scratch/rss230/sharma_choi_images/realsim_images_071321/"
image_files = pathlib.Path(image_loc).glob("*.fits")
out_file = "/scratch/rss230/AGN-Mergers/data/morphological_fits.h5"

data = []

def calc_morphology(fname):
    image_name = fname.name

    halo_num = int(image_name.split('.')[0][1:])
    timestep = int(image_name.split('.')[1])

    morph_values = np.full(len(morph_params), np.nan)
    try:
        source_morph = am.source_morphology(
            fname, input_ext_name="RealSim_Smooth",
        )  # Calculate morphological parameters using statmorph

        if source_morph is not None:
            morph_values = [source_morph[value] for key, value in morph_params.items()]
        print(image_name)
        return [halo_num, timestep, *morph_values]

    except (KeyError, IndexError, AttributeError, ValueError, TypeError) as err:
        print(err, "-", image_name, "not processed, skipping fit.")
        return [halo_num, timestep, *morph_values]

data = [calc_morphology(fname) for fname in image_files]

df = pd.DataFrame(np.asarray(data), columns=['halo_num', 'timestep', *list(morph_params.keys())])
df.to_hdf(out_file, key='run2')