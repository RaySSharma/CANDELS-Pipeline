#code for converting pristine simulated images to realistic mocks
import astropy
import astropy.io.ascii as ascii
import astropy.io.fits as fits
import numpy as np
#np.random.standard_normal(g.shape)
import matplotlib
import matplotlib.pyplot as pyplot
import astropy.units as u
import glob
import sys
import os

import photutils
from astropy.stats import gaussian_fwhm_to_sigma

import scipy.ndimage
import scipy as sp
#res=sp.ndimage.filters.gaussian_filter(b,sigma,output=sB)
#import astropy.convolution.convolve as convolve


#modify filename and metadata from input image, if desired
def output_pristine_fits_image(original_image,out_image):

    return


def convolve_with_fwhm(in_image, out_image, fwhm_arcsec=0.10):

    #open pristine image fits file
    in_fo=fits.open(in_image,'readonly')

    #load image data and metadata
    image_in=in_fo['SimulatedImage'].data
    header_in=in_fo['SimulatedImage'].header

    #calculate PSF width in pixel units
    pixel_size_arcsec=header_in['PIXSCALE']
    sigma_arcsec=fwhm_arcsec*gaussian_fwhm_to_sigma
    sigma_pixels=sigma_arcsec/pixel_size_arcsec
    
    image_out=sp.ndimage.filters.gaussian_filter(image_in,sigma_pixels,mode='nearest')
    hdu_out = fits.ImageHDU(image_out,header=header_in)
    hdu_out.header['FWHMPIX']=(sigma_pixels/gaussian_fwhm_to_sigma,'pixels')
    hdu_out.header['FWHM']=(fwhm_arcsec,'arcsec')
    hdu_out.header['SIGMAPIX']=(sigma_pixels,'pixels')
    hdu_out.header['SIGMA']=(sigma_arcsec,'arcsec')
    hdu_out.header['EXTNAME']='MockImage_Noiseless'

    
    return


def add_simple_noise(in_image, out_image, sb=25.0):

    return
