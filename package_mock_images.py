import glob
import os
import numpy as np
import pandas as pd
import astropy.io.fits as fits
from re import findall

"""Package images for sharing and output a data file of BH, halo, and morphological parameters.
"""

SB_LIMIT = "25"
OVERWRITE_IMAGES = True
PACKAGE_MOCK_IMAGES = True
OUTPUT_PARAMETERS = True
NUM_RUNS = 1
BH_MODEL = "Hopkins"
SIMULATION_DIR = "/projects/somerville/GADGET-3/Fiducial_Models/Fiducial_withAGN_hdf/"
IMAGE_INPUT_DIR = "/scratch/rss230/AGN-Obscuration/outputs/*/*/WFC3_F160W/[3-5]/"
PACKAGED_IMAGE_OUTPUT_DIR = "/scratch/rss230/sharma_choi_images/orig_images/"
PARAMETER_OUTPUT_FILE = (
    "/home/rss230/AGN-Obscuration/outputs/data_SB" + SB_LIMIT + ".h5"
)
MORPH_PARAMS = {
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
TOTAL_MISSING_BH_SED = 0


def save_hdu(hdu, filename):
    hdu.writeto(filename, overwrite=OVERWRITE_IMAGES)
    print("Output image:", filename)


def gather_bh_lum(im):
    """Gather BH properties from Powderday bh_sed.npz file
    
    Arguments:
        im {str} -- Image filename 
    
    Returns:
        luminosity [float] -- Total BH luminosity
    """
    global TOTAL_MISSING_BH_SED

    try:
        with np.load(os.path.dirname(im) + "/bh_sed.npz") as bh_sed:
            lum = bh_sed["luminosity"].sum()
            # fnu = bh_sed['fnu']
            # nu = bh_sed['nu']
            # lum_obsc = -np.sum(np.trapz(fnu / nu, nu, axis=1))
    except FileNotFoundError:
        print("No bh_sed.npz found.")
        TOTAL_MISSING_BH_SED += 1
        lum = np.nan
        # lum_obsc = np.nan
    return lum


def gather_halo_properties(halo_num, timestep):
    """Gather halo properties from simulation info file
    
    Arguments:
        halo_num {str} -- Halo number
        timestep {str} -- Simulation snapshot
    
    Returns:
        Mstar {str} -- Stellar mass
        M200 {str} -- Halo mass
        Mgas {str} -- Total gas mass
    """
    info_file = SIMULATION_DIR + halo_num + "/info_" + timestep + ".txt"
    with open(info_file) as f:
        lines = f.readlines()
        try:
            Mstar = findall("[0-9].[0-9]*e\+[0-9]*", lines[17])[0]
            M200 = findall("[0-9].[0-9]*e\+[0-9]*", lines[11])[0]
            Mgas = findall("[0-9].[0-9]*e\+[0-9]*", lines[18])[0]
        except IndexError:
            print("No halo parameters found.")
            Mstar = np.nan
            M200 = np.nan
            Mgas = np.nan
    return Mstar, M200, Mgas


def gather_sim_properties(im):
    """Gather simulation properties obtainable from image filename
    
    Arguments:
        im {str} -- Image filename
    
    Returns:
        halo_num {str} -- Halo number
        timestep {str} -- Simulation snapshot
        filter_name {str} -- Filter used for imaging
        SB {str} -- Surface brightness in mag arcsec^-2
    """
    halo_properties = im.split("/")[-1].split(".")
    halo_num = halo_properties[0]
    timestep = halo_properties[1]
    filter_name = halo_properties[2]
    SB = halo_properties[4]

    return halo_num, timestep, filter_name, SB


def gather_data(im, morph, z):
    """Gather up data for exporting
    
    Arguments:
        im {str} -- Image filename
        morph {object} -- Source morphology object
        z {float} -- Redshift
        bh_model {str} -- Black hole model type
    
    Returns:
        np.array -- Data to export
    """
    lum = gather_bh_lum(im)
    halo_num, timestep, filter_name, SB = gather_sim_properties(im)
    morph_params = gather_morph_params(morph)
    Mstar, M200, Mgas = gather_halo_properties(halo_num, timestep)
    image_data = [
        halo_num,
        timestep,
        z,
        filter_name,
        SB,
        BH_MODEL,
        lum,
        Mstar,
        M200,
        Mgas,
        morph_params,
    ]
    return image_data


def gather_morph_params(morph):
    params = []
    for par in MORPH_PARAMS:
        try:
            params.append(morph[par])
        except KeyError:
            params.append(np.nan)
    return params


image_files = glob.glob(IMAGE_INPUT_DIR + "*.image.SB" + SB_LIMIT + ".fits")

packaged_data = []
for i, image in enumerate(image_files):
    print("Image:", image)
    print("Image number:", i, "/", len(image_files))
    with fits.open(image) as f:
        image_number = str(int(image.split("/")[-2]) - 3)
        packaged_filename = (
            PACKAGED_IMAGE_OUTPUT_DIR
            + os.path.basename(image)[:-5]
            + "."
            + image_number
            + ".fits"
        )

        if PACKAGE_MOCK_IMAGES:
            data = f["MockImage"].data
            header = f["MockImage"].header
            morphology = f["SOURCE_MORPH"].header
            redshift = header["REDSHIFT"]
            gathered_data = gather_data(image, morphology, redshift)

            header["MSTAR"] = (float(gathered_data[7]), "Msol")
            packaged_data.append(gathered_data)
        else:
            data = f["SimulatedImage"].data
            header = f["SimulatedImage"].header

        f_out = fits.PrimaryHDU(data=data, header=header)
        save_hdu(f_out, packaged_filename)

if OUTPUT_PARAMETERS and PACKAGE_MOCK_IMAGES:
    columns = [
        "halo_num",
        "timestep",
        "redshift",
        "filter",
        "SB",
        "bh_model",
        "Lbol",
        "Mstar",
        "M200",
        "Mgas",
    ] + list(MORPH_PARAMS.keys())
    packaged_data = np.array(packaged_data)
    df2 = pd.DataFrame(packaged_data, columns=columns)

    if os.path.isfile(PARAMETER_OUTPUT_FILE):
        df1 = pd.read_hdf(PARAMETER_OUTPUT_FILE, key="data")
    else:
        df1 = pd.DataFrame()

    df = pd.concat([df1, df2], sort=True)
    df.to_hdf(PARAMETER_OUTPUT_FILE, key="data")

