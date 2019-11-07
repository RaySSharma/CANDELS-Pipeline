import astropy.io.fits as fits
import glob
import os

BH_model = 'Nenkova'
SB_limit = '25'

image_loc = '/home/rss230/AGN-Obscuration/outputs/BH+' + BH_model + '/*/*/WFC3_F160W/*/'
image_files = glob.glob(image_loc + '*.image.SB' + SB_limit + '.fits')
output_dir = '/scratch/rss230/Kocevski_images/'

for i, image in enumerate(image_files):
    print('Image:', image)
    print('Image number:', i, '/', len(image_files))
    with fits.open(image) as f:
        output_filename = output_dir + os.path.basename(image)
        print('Output image:', output_filename)
        data = f['MockImage'].data
        header = f['MockImage'].header
        try:
            fits.writeto(output_filename, data, header)
        except IOError:
            fits.update(output_filename, data, header)
