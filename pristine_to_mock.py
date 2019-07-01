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

    return


def add_simple_noise(in_image, out_image, sb=25.0):

    return
