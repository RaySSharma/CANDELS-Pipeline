import glob
import os
import tqdm
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
SIMULATION_DIR = "/scratch/rss230/GADGET-3/Fiducial_Models/Fiducial_withAGN_hdf/"
IMAGE_INPUT_DIR = "/scratch/rss230/sharma_choi_images/realsim_images_071321/"
PARAMETER_OUTPUT_FILE = (
    "/scratch/rss230/AGN-Mergers/outputs/data_SB" + SB_LIMIT + ".h5_082421"
)
BH_SED_DIR = "/scratch/rss230/AGN-Mergers/outputs/"
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
TOTAL_MISSING_BH_SED = 0
BAD_VAL = -99


def save_hdu(hdu, filename):
    hdu.writeto(filename, overwrite=OVERWRITE_IMAGES)
    print("Output image:", filename)


def gather_bh_lum(halo_num, timestep, orientation):
    """Gather BH properties from Powderday bh_sed.npz file
    
    Arguments:
        im {str} -- Image filename 
    
    Returns:
        luminosity [float] -- Total BH luminosity
    """
    global TOTAL_MISSING_BH_SED

    try:
        im_num = str(int(orientation) + 3)
        bh_file = (
            BH_SED_DIR
            + halo_num
            + "/"
            + timestep
            + "/"
            + "WFC3_F160W/"
            + im_num
            + "/bh_sed.npz"
        )
        with np.load(bh_file) as bh_sed:
            lum = bh_sed["luminosity"].sum()
            # fnu = bh_sed['fnu']
            # nu = bh_sed['nu']
            # lum_obsc = -np.sum(np.trapz(fnu / nu, nu, axis=1))
    except FileNotFoundError:
        print("No bh_sed.npz found.")
        TOTAL_MISSING_BH_SED += 1
        lum = BAD_VAL
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
    try:
        with open(info_file) as f:
            lines = f.readlines()
            Mstar = findall("[0-9].[0-9]*e\+[0-9]*", lines[17])[0]
            M200 = findall("[0-9].[0-9]*e\+[0-9]*", lines[11])[0]
            Mgas = findall("[0-9].[0-9]*e\+[0-9]*", lines[18])[0]
    except (IndexError, FileNotFoundError) as err:
        print("No halo parameters found.")
        Mstar = BAD_VAL
        M200 = BAD_VAL
        Mgas = BAD_VAL
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
    orientation = halo_properties[5]

    return halo_num, timestep, filter_name, SB, orientation


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
    halo_num, timestep, filter_name, SB, orientation = gather_sim_properties(im)
    lum = gather_bh_lum(halo_num, timestep, orientation)
    morph_params = gather_morph_params(morph)
    Mstar, M200, Mgas = gather_halo_properties(halo_num, timestep)
    image_data = [
        halo_num,
        timestep,
        z,
        filter_name,
        SB,
        os.path.basename(im),
        lum,
        Mstar,
        M200,
        Mgas,
        *morph_params,
    ]
    return image_data


def gather_morph_params(morph):
    params = []
    for par in MORPH_PARAMS:
        try:
            params.append(morph[par])
        except KeyError:
            params.append(BAD_VAL)
    return params


image_files = glob.glob(IMAGE_INPUT_DIR + "*.fits")

packaged_data = []
for i, image in enumerate(tqdm.tqdm(image_files)):

    with fits.open(image) as f:
        if PACKAGE_MOCK_IMAGES:
            hdu0 = fits.PrimaryHDU()
            hdu3 = f["SimulatedImage"]

            try:
                morphology = f["SOURCE_MORPH"].header
            except:
                print("No SOURCE_MORPH header")
                continue
            redshift = hdu3.header["REDSHIFT"]
            gathered_data = gather_data(image, morphology, redshift)
            packaged_data.append(gathered_data)

if OUTPUT_PARAMETERS and PACKAGE_MOCK_IMAGES:
    columns = [
        "halo_num",
        "timestep",
        "redshift",
        "filter",
        "SB",
        "filename",
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
