#!/usr/bin/env python3
import numpy as np
import pristine_to_mock as ptm
import analyze_mock as am
import glob
import matplotlib

matplotlib.use("Agg")
from mpi4py import MPI

DEBLEND = True
IMAGE_LOC = "/scratch/rss230/AGN-Obscuration/outputs/*/*/WFC3_F160W/[3-5]/"


# {filter: central wavelength [micron], instrument resolution [arcsec], gain}
filt_wheel = {
    "ACS_F814W": [0.8353, 0.124, 1.55],
    "WFC3_F160W": [1.5369, 0.145, 2.4],
    "ACS_F606W": [0.5907, 0.124, 1.55],
}
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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    image_files = glob.glob(IMAGE_LOC + "*.image.fits")
    num_images = len(image_files)
    images_per_proc = int(np.ceil(num_images / size))
    data = []
    for i in range(size):
        try:
            data.append(image_files[i * images_per_proc : (i + 1) * images_per_proc])
        except IndexError:
            pass
else:
    data = None

data = comm.scatter(data, root=0)

for i, image in enumerate(data):
    print("[PROC " + str(rank) + "]:", i, "/", len(data), "Image:", image, flush=True)
    for lim in detection_limits:
        image_mock = image[:-5] + ".SB" + str(lim) + ".fits"
        fig_name = image[:-5] + ".SB" + str(lim) + ".png"
        ext_name = "MockImage"

        print("[PROC " + str(rank) + "]:", "Output Image:", image_mock, flush=True)

        ptm.output_pristine_fits_image(
            image, image_mock, filt_wheel
        )  # Setup image for mock creation
        ptm.convolve_with_fwhm(image_mock, filt_wheel)  # Convolve mock image with PSF
        ptm.add_simple_noise(
            image_mock, sb_maglim=lim, ext_name=ext_name, alg="Snyder2019"
        )  # Add noise model

        seg, kernel, errmap, bkg = am.detect_sources(
            image_mock, ext_name=ext_name, filt_wheel=filt_wheel
        )  # Run source detection with photutils

        if DEBLEND:
            seg = am.deblend_sources(
                image_mock, seg, kernel, errmap, ext_name=ext_name
            )  # Deblend detected sources
            props_ext_name = "DEBLEND_PROPS"
        else:
            props_ext_name = "SEGMAP_PROPS"

        source_morph = am.source_morphology(
            image_mock,
            seg,
            errmap=errmap,
            bkg=bkg,
            ext_name=ext_name,
            props_ext_name=props_ext_name,
        )  # Calculate morphological parameters using statmorph

        am.save_morph_params(
            image_mock, source_morph, fig_name=fig_name, **morph_params
        )  # Save morph params to HDU, generate statmorph image of params

finished = comm.gather(data, root=0)

if rank == 0:
    print("Finished successfully")

