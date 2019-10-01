#!/usr/bin/env python3
import glob
import pristine_to_mock as ptm
import analyze_mock as am

DEBLEND = True

# {filter: central wavelength [micron], instrument resolution [arcsec], gain}
filt_wheel = {
    'ACS_F814W': [0.8353, 0.05, 1.55],
    'WFC3_F160W': [1.5369, 0.13, 2.4],
    'ACS_F606W': [0.5907, 0.05, 1.55]
}
# mag/arcsec^2
detection_limits = [25, 27]

# List of morphological parameters to calculate in statmorph
morph_params = {
    'GINI': 'gini',
    'M20': 'm20',
    'CONC': 'concentration',
    'ASYM': 'asymmetry',
    'SMOOTH': 'smoothness',
    'SERSIC_N': 'sersic_n',
    'M': 'multimode',
    'D': 'deviation',
    'I': 'intensity'
}

#image_loc = '/home/rss230/AGN-Obscuration/outputs/BH+Hopkins/*/*/ACS_F814W/1/'
image_loc = './data/'
image_files = glob.glob(image_loc + '*.image.fits')

for image in image_files:
    print('Image:', image)
    for lim in detection_limits:
        image_mock = image[:-5] + '.SB' + str(lim) + '.fits'
        fig_name = image[:-5] + '.SB' + str(lim) + '.png'
        ext_name = 'MockImage'

        print('SB:', lim, 'mag arcsec^-2')
        print('Mock Image:', image_mock)

        ptm.output_pristine_fits_image(
            image, image_mock, filt_wheel)  # Setup image for mock creation
        ptm.convolve_with_fwhm(image_mock,
                               filt_wheel)  # Convolve mock image with PSF
        ptm.add_simple_noise(image_mock, sb_maglim=lim, ext_name=ext_name,
                             alg='Snyder2019')  # Add noise model

        seg, kernel, errmap = am.detect_sources(
            image_mock,
            ext_name=ext_name)  # Run source detection with photutils

        if DEBLEND:
            seg = am.deblend_sources(
                image_mock, seg, kernel, errmap,
                ext_name=ext_name)  # Deblend detected sources
            props_ext_name = 'DEBLEND_PROPS'
        else:
            props_ext_name = 'SEGMAP_PROPS'

        source_morph = am.source_morphology(
            image_mock, seg, filt_wheel, ext_name=ext_name,
            props_ext_name=props_ext_name
        )  # Calculate morphological parameters using statmorph

        am.save_morph_params(
            image_mock, source_morph, fig_name=fig_name, **morph_params
        )  # Save morph params to HDU, generate statmorph image of params
