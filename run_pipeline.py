import glob

# {filter: central wavelength [micron], instrument resolution [arcsec]}
filt_wheel = {
    'ACS_F814W': [0.8353, 0.05],
    'WFC3_F160W': [1.5369, 0.13],
    'ACS_F606W': [0.5907, 0.05]
}

image_loc = '/home/rss230/AGN-Obscuration/outputs/BH+Hopkins/*/*/ACS_F814W/1/'
image_files = glob.glob(image_loc+'*.image.fits')

for image in image_files:
    ffout = image[:-5]+'.mock.fits'
    output_pristine_fits_image(ff, ffout, filt_wheel)

    convolve_with_fwhm(ffout, filt_wheel)
    add_simple_noise(ffout, sb_maglim=25.0, sb_label='25', alg='Snyder2019')
    add_simple_noise(ffout, sb_maglim=27.0, sb_label='27', alg='Snyder2019')



