import glob
import os
import numpy as np
import pandas as pd
import astropy.io.fits as fits

SB_LIMIT = '25'
BH_MODEL = 'Nenkova'
OVERWRITE_IMAGES = True
MOCK = True

IMAGE_LOC = '/scratch/rss230/AGN-Obscuration/outputs/*/*/WFC3_F160W/0/'
IMAGE_FILES = glob.glob(IMAGE_LOC + '*.image.SB' + SB_LIMIT + '.fits')
IMAGE_OUTPUT_DIR = '/scratch/rss230/Kocevski_images/'
DATA_OUTPUT = '/home/rss230/AGN-Obscuration/outputs/data_SB' + SB_LIMIT + '.h5'
MORPH_PARAMS = {
    'GINI': 'gini',
    'M20': 'm20',
    'CONC': 'concentration',
    'ASYM': 'asymmetry',
    'SMOOTH': 'smoothness',
    'SERSIC_N': 'sersic_n',
    'M': 'multimode',
    'D': 'deviation',
    'I': 'intensity',
    'FLAG': 'flag',
    'FLAG_SERSIC': 'flag_sersic'
}
MISSING_BH_SED = 0


def save_hdu(hdu, filename):
    count = 0
    while True:
        filename = filename[:-7] + '.' + str(count) + '.fits'
        try:
            hdu.writeto(filename, overwrite=OVERWRITE_IMAGES)
            break
        except OSError:
            count += 1
    print('Output image:', filename)


def gather_data(im, morph, z):
    global MISSING_BH_SED
    try:
        with np.load(os.path.dirname(im) + '/bh_sed.npz') as bh_sed:
            lum = bh_sed['luminosity'].sum()
            fnu = bh_sed['fnu']
            nu = bh_sed['nu']
            lum_obsc = -np.sum(np.trapz(fnu / nu, nu, axis=1))
    except FileNotFoundError:
        print('No bh_sed.npz found.')
        MISSING_BH_SED += 1
        lum = np.nan
        lum_obsc = np.nan

    halo_properties = im.split('/')[-1].split('.')
    halo_num = halo_properties[0]
    timestep = halo_properties[1]
    filter_name = halo_properties[2]
    SB = halo_properties[4]
    morph_params = gather_morph_params(morph)
    image_data = np.hstack([
        halo_num, timestep, z, filter_name, SB, BH_MODEL, lum, lum_obsc,
        morph_params
    ])
    return image_data


def gather_morph_params(morph):
    params = []
    for par in MORPH_PARAMS:
        try:
            params.append(morph[par])
        except KeyError:
            params.append(np.nan)
    return params


packaged_data = []
for i, image in enumerate(IMAGE_FILES):
    print('Image:', image)
    print('Image number:', i, '/', len(IMAGE_FILES))
    with fits.open(image) as f:
        output_filename = IMAGE_OUTPUT_DIR + os.path.basename(
            image)[:-5] + '.0.fits'
        if MOCK:
            data = f['MockImage'].data
            header = f['MockImage'].header
            morphology = f['SOURCE_MORPH'].header
            redshift = header['REDSHIFT']
            packaged_data.append(
                gather_data(image, morphology,
                            redshift))  # Gather morphological + halo + BH data
        else:
            data = f['SimulatedImage'].data
            header = f['SimulatedImage'].header
            redshift = header['REDSHIFT']

        f_out = fits.PrimaryHDU(data=data, header=header)
        save_hdu(f_out, output_filename)  # Save HDU

if not MOCK:
    columns = [
        'halo_num', 'timestep', 'redshift', 'filter', 'SB', 'BH_model', 'Lbol',
        'Lbol_obsc'
    ] + list(MORPH_PARAMS.keys())
    packaged_data = np.array(packaged_data)
    df2 = pd.DataFrame(packaged_data, columns=columns)

    if not os.path.isfile(DATA_OUTPUT):
        df1 = pd.DataFrame()
    else:
        df1 = pd.read_hdf(DATA_OUTPUT, key='data')

    df = pd.concat([df1, df2], sort=True)
    df.to_hdf(DATA_OUTPUT, key='data')
