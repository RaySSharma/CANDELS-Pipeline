#!/usr/bin/env python3
import pristine_to_mock as ptm
import analyze_mock as am
import RealSim.ObsRealism as obs
import glob
import matplotlib

matplotlib.use("Agg")

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

        ptm.output_pristine_fits_image(
            image, image_mock, filt_wheel
        )  # Setup image for mock creation
        ptm.convolve_with_fwhm(image_mock, filt_wheel)  # Convolve mock image with PSF
        ptm.add_simple_noise(
            image_mock, sb_maglim=lim, alg="Snyder2019"
        )  # Add noise model

        candels_args = obs.make_candels_args(field_info, realsim_input_dir)
        obs.ObsRealism(
            image_mock, candels_args
        )  # Apply RealSim CANDELS fields

        seg, kernel, errmap = am.detect_sources(
            image_mock, ext_name="RealSim", filt_wheel=filt_wheel
        )  # Run source detection with photutils

        seg = am.deblend_sources(
            image_mock, seg, kernel, errmap, ext_name="RealSim"
        )  # Deblend detected sources
