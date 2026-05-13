#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 21:29:10 2025

@author: ciarachisholm
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time


import parameters_file as pf
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
import scipy.optimize as opt
import scipy.special as sp
from pathlib import Path
import os
import copy
# from main_code.subroutines import fits_handling as fh
# from main_code.subroutines import misc_functions as mf
# from main_code.subroutines import array_calculations as ac
from os import makedirs
import sys

from main_code.subroutines import fits_handling as fh
from main_code.subroutines import misc_functions as mf
from main_code.subroutines import array_calculations as ac

def PIimg(mosaic="ma1", plot=True, return_StoN=False, gal_coord=True, click_coord=False):

    """This function plots the polarized intensity of a CGPS mosaic. It assumes that matplotlib and 
    numpy have been imported into the script, and that astropy has been installed on the device.
    This function also prints the pixel coordinates of where the cursor is clicked. 
    
    Parameters:
        mosaic (string): The name of mosaic to be shown (default: ma1)
        
        plot (boo): Whether or not to plot the function (Default = true)
        
        return_StoN (boo): Whether or not to return the signal to noise of image
        
    Returns:
        PI_cor: the corrected Polarized intensity of the mosaic
        
        SignToNoise (optional): the signal to noise of each pixel in the mosaic if selected
        """
    from astropy.io import fits
    from astropy.wcs import WCS
    import parameters_file as pf
    
    # plt.rcParams.update({'font.size': 16})
    # Creating a list of the frequency bands to loop through
    bands = ["A","B", "C", "D"]

    # Creating the empty lists to store the image information
    imlistQ = []
    imlistU = []
    imlistw = []
    #creating an empty list to store the uncertainty of the Stokes parameters δQ_ij and δU_ij
    sigma_squared = []
    sigmas = []
    #Creating a list to store the uncorrected polarised intensities;
    PIs = []
    obsfreq =[]

    # Reading the fits file
    hdu_listI = fits.open(pf.img_dir  +mosaic+"_1420_MHz_I_image.fits")
    
    #Getting the mosaic information for the coordinates 
    headerI = hdu_listI[0].header
    # print(repr(headerI))
    # getting and removing unnecessary dimensions from the data
    imI = np.squeeze(hdu_listI[0].data)
    
    #setting the nuq array
    nuq = np.full(imI[0].shape, 0.00028) # the rms noise here is based on JB thesis
    nuq2 = nuq**2
    # Setting the fraction of stokes I to be included in the PI error
    alpha=0.003
    
    #looping through each band
    for band in bands:
        #getting the path to each band
        pathQ = pf.img_dir   +mosaic+"_1420_MHz_" +"Q_" +band+"_image.fits"
        pathU = pf.img_dir   +mosaic+"_1420_MHz_" +"U_" +band+"_image.fits"
        pathw = pf.img_dir   +mosaic+"_1420_MHz_POL_" +band+"_wght.fits"
        
        
        #opening the files 
        hdu_listQ = fits.open(pathQ)
        hdu_listU = fits.open(pathU)
        hdu_listw = fits.open(pathw)
        
        #getting the data from the file
        image_dataQ = hdu_listQ[0].data
        image_dataU = hdu_listU[0].data
        image_dataw = hdu_listw[0].data
        
        obsfreq.append(hdu_listQ[0].header["OBSFREQ"]/1000000)
        
        # if band=="D":
        #     print(repr(hdu_listQ[0].header ))
        # removing unnecessary dimensions from the data
        imQ = np.squeeze(image_dataQ)
        imU = np.squeeze(image_dataU)
        imw = np.squeeze(image_dataw)
        
        #storing the information of each image 
        imlistQ.append(imQ)
        imlistU.append(imU)
        imlistw.append(imw)
        
        #Adding the uncertainty of the band to the designated list 
        sigmas.append(np.sqrt((nuq2/imw)+(alpha*imI)**2))
        sigma_squared.append((nuq2/imw)+(alpha*imI)**2)

        #Adding the polarised intensity of the band to the list
        PIs.append(np.sqrt(imQ**2+imU**2))
    
    # print(hdu_listQ[0].header["OBSFREQ"]/1000000)
    # print(obsfreq)
    #Converting to arrays
    PIs = np.array(PIs)
   
    sigma_squared = np.array(sigma_squared)
    sigmas = np.array(sigmas)
    
    # print(np.shape(PIs))
    #Calculating noise bias and variability 
    noisebias= np.sqrt(2)*np.sum(sigmas, axis=0)/4 # adding all the 
    noisevar = np.sqrt(sigma_squared[0]**1 +sigma_squared[1]**1+sigma_squared[2]**1+sigma_squared[3]**1)/4 

    # Calculating the corrected polarised intensity. 
    PI_cor = np.sum(PIs, axis=0)/4 - noisebias
    
    SignToNoise= (PI_cor)/noisevar
    # Calculating the average weight over all the channels
    av_wght = (imlistw[0] + imlistw[1] + imlistw[2] + imlistw[3])/4
    
    
    
    #getting the decliation of the moasic (degrees)
    dec = headerI[64]
    # print(mosaic, " dec: ", dec)
    
   
    if plot == True:
        
        
        #Plotting the image
    
        fig, ax = plt.subplots(1,1,dpi = pf.DPI, figsize=(8,8))

        # plt.figure(dpi = pf.DPI, figsize=(8,8))
        ax.set_title("The Polarised Intensity of "+mosaic.upper(), fontsize=30)
        if click_coord:
            def mouse_event(event):
                print('x: {} and y: {} in pixel coordinates'.format(np.round(event.xdata,2), np.round(event.ydata)))
            
            cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
        
        
        #making an array with the number of pixels in the image
        ticksx = np.linspace(0, len(PI_cor[0]), len(PI_cor[0]))
        ticksy = np.linspace(0, len(PI_cor[:,0]), len(PI_cor[0]))
        # Setting the number of ticks to be displayed on the plot
        # tck = [n for n in range(0,1024,pf.PI_num_of_pixels_btw_ticks)]
        
        tck = [n for n in range(0,len(PI_cor[0]),1)]
        if gal_coord:
            # Adding the galactic coordinates to the image, the coordinates will 
            # not change between the files so any header can be used for this 
            w = WCS(headerI)
            
            
            #Using the information from the header and the number of pixels determining 
            # what the coordinates of the image is 
            wx, wy, f, meh = w.all_pix2world(ticksx, ticksy,0,0,1)
            
            
            # Getting the labels of the x and y ticks
            tickx_labels = np.round(wx[tck],2)
            ticky_labels = np.round(wy[tck],2)
            
            
            ax.set_xticks(tck, tickx_labels,  fontsize=12)
            ax.set_yticks(tck, ticky_labels,  fontsize=12)
            
 
            ax.set_xlabel(r"Longitude ($\degree$)",  fontsize=25)
            ax.set_ylabel (r"Latitude ($\degree$)",  fontsize=25)
            
        else:
            plt.xticks(tck)
            plt.yticks(tck)
        # print(pf.PI_VMAX)
        PLT = ax.imshow(PI_cor, vmin=pf.PI_VMIN, vmax=pf.PI_VMAX,cmap="gist_heat",origin='lower',)
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)
        
        cbar = fig.colorbar(PLT,cax=cax, )
        cbar.set_label(label="Polarised Intensity (mJy/beam)", size=20)
        ticksforcbar = np.linspace( pf.PI_VMIN, pf.PI_VMAX ,7)
        cbar.set_ticks(ticksforcbar.tolist())
        # cbar.ax.tick_params(labelsize=12)
        labels_for_colorbar = np.round(np.copy(ticksforcbar)*1000,1)
        cbar.ax.set_yticklabels(labels_for_colorbar.tolist())

        
        # ax.ticker.MaxNLoactor(nbins="auto")
        ax.xaxis.set_major_locator(plt.MaxNLocator("auto"))
        ax.yaxis.set_major_locator(plt.MaxNLocator("auto"))
        plt.draw()
        plt.tight_layout()
        plt.show()
    
    if return_StoN == False:
        return PI_cor
    else:
        return PI_cor,  SignToNoise

def read_qu_data_cve(directory_path, mosaic_name):
    """This function reads data and header information from fits images from a particular mosaic.
    
    Returns:
        stokes (dict): a dictionary containing all the Stokes images for each band 
            including Stokes I, Q_A, Q_B, Q_C, Q_D, U_A, U_B, U_C, U_D
        header (dict): a dictionary that contains the header for each of the 
            Stokes images. 
    """
    stokes = {}
    header = {}

    for band in ['I', 'Q_A', 'Q_B', 'Q_C', 'Q_D', 'U_A', 'U_B', 'U_C', 'U_D']:
        stokes_header, stokes_data = fh.read_fits(f'{directory_path}{mosaic_name}/m{mosaic_name}_1420_MHz_{band}_image.fits')

        stokes[band] = stokes_data
        header[band] = stokes_header
        
    
    
    return stokes, header
def RM_code_StoN( Mo, y, x, mosaic_directory=None, thres = 5, return_surrounding_pixels_above_threshold=False):
    """Returns the Signal to noise array that the RM code produces. 
    if return_surrounding_pixels then returns the StoN of (y,x) and a bool about 
    whether the surrounding (top, bottom, left, right) pixels are above the threshold.
    if not, it just returns the StoN of the source. """
    
    
    
    
    PIim1 = PIimg(Mo, plot=False)
    
    stokes_i_threshold = 1.2 / 1000
    min_pol_threshold = 0.02
    alpha = 0.003  # fraction of I to be included in PA error
    output_ext = 'final_003I'
    edge_threshold = 5.0
    fwxm_v = 2.0  
    fwxm = np.sqrt(2.0 * np.log(fwxm_v))
    
    box_halfwidth = 20
    pi_units = 'mJy/beam'
    
    if Mo[0] == "m" or Mo[0]=="M":
        mosaic_name = Mo[1:]
    else:
        mosaic_name=Mo
    
    if mosaic_directory==None:
        onedrive_path = os.path.dirname(os.path.realpath(__file__))[:-5]
        
        if onedrive_path[1:5] =="home":
            INPUT_PATH = '''/home/ciara.chisholm1/OneDrive/Ciara's Research Cubby/Codes/rmap-main1/Data_f/raw_data/'''
            # img_dir = "/scratch/CGPSData/CGPS2012/"
        elif onedrive_path[1:5] =="User":
            # img_dir = "/Users/ciarachisholm/Desktop/Research/CGPS2012/"
            INPUT_PATH = '''/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Codes/rmap-main1/Data_f/raw_data/'''
        else:
            print("WARNING: Image directory not configure for the current device. Please add image directory option to the directories file. "+\
                  "The data should ideal be stored locally to avoid a bottle neck when loading the datafiles.")
                
    #Checking which computer 
    
    input_directory = INPUT_PATH
    stokes, header = read_qu_data_cve(input_directory, mosaic_name)
    from sys import exit
    
    x_array, y_array, x_long, y_lat = fh.make_xy_arrays(header['I'])
    


    xpix_max_i =x # Getting the x coordinate of maximum intensity in pixel values for the source 
    ypix_max_i = y # Getting the y coordinate of maximum intensity in pixel values for the source 
    
    
    
    # Because the box sizes are different between the noise calculation and the Gauss fit,
    # it's convenient to define a full-size PI array, fill part of it with the FG-subtracted and
    # de-biased PI, then feed that into the old code so it can extract its own box.
    # Same for S:N array and noise
    temp_pi = np.zeros(x_array.shape) # I am assuming this is a an array containing a copy of the PI data- Ciara Chisholm 
    ston = np.zeros(x_array.shape) # Creating an array to store the Signal to Noise of the image
    noise = np.zeros(x_array.shape) # Array to store the noise  
    
    x_min = mf.nround(xpix_max_i) - box_halfwidth
    x_max = mf.nround(xpix_max_i) + box_halfwidth + 1 # The + 1s are because IDL indexes arrays differently than python
    y_min = mf.nround(ypix_max_i) - box_halfwidth
    y_max = mf.nround(ypix_max_i) + box_halfwidth + 1 # The + 1s are because IDL indexes arrays differently than python

    stamp = {}
    
    # Ciara Chisholm October 11 2024 modified the arrays to include the copy of the original arrays.
    OG_arrays = [x_array, y_array, stokes['I'], stokes['Q_A'], stokes['Q_B'],
              stokes['Q_C'], stokes['Q_D'], stokes['U_A'], stokes['U_B'], stokes['U_C'], stokes['U_D']]
    arrays = OG_arrays.copy()
    array_label = ['xarr', 'yarr', 'I', 'Q_A', 'Q_B', 'Q_C', 'Q_D', 'U_A', 'U_B', 'U_C', 'U_D']
    
    for array in range(len(arrays)):
        # Note: argument 2-5 are the same as x_min, x_max, y_min, y_max. 
        #   The +1 is added in the cut_out function. 
        stamp[array_label[array]] = fh.cut_out_stamp(arrays[array],
                                                     mf.nround(xpix_max_i) - box_halfwidth,
                                                     mf.nround(xpix_max_i) + box_halfwidth,
                                                     mf.nround(ypix_max_i) - box_halfwidth,
                                                     mf.nround(ypix_max_i) + box_halfwidth)
        # The stamps have been stored in a dictionary, just like the fits data was. To access simply call, for example, stamp['xarr'], or stamp['I']
        
        
    from astropy.io import fits
    from astropy.wcs import WCS   
    
    hdu_list = fits.open(pf.img_dir  +Mo+"_1420_MHz_I_image.fits")
    HEADER = hdu_list[0].header
    
    w = WCS(HEADER)
    
    ticksx = np.linspace(0, len(PIim1[0]), len(PIim1[0]))
    ticksy = np.linspace(0, len(PIim1[:,0]), len(PIim1[0]))
    
    wx, wy, f, meh = w.all_pix2world(ticksx, ticksy,0,0,1)
    L = wx[x]
    B = wy[y]
    # Noise calculations:
    # The main purpose of this is to generate the noise-arr. The foreground subtraction isn't used here.
    g_coords = SkyCoord(l=L, b=B, frame='galactic', unit='degree')
    ra = g_coords.fk5.ra.deg
    dec = g_coords.fk5.dec.deg

    annulus_pixels = ac.calculate_annulus(ra, dec, stamp['xarr'], stamp['yarr'], xpix_max_i, ypix_max_i, stamp['I'], stokes_i_threshold)
    foreground_pixels = annulus_pixels[0]


    
    # foreground contains an array of noise for each channel, sigma_qu is mean noise of all channels. 
    foreground_vector, sigma_qu = ac.estimate_local_noise(foreground_pixels,
                                                          stamp['Q_A'],
                                                          stamp['Q_B'],
                                                          stamp['Q_C'],
                                                          stamp['Q_D'],
                                                          stamp['U_A'],
                                                          stamp['U_B'],
                                                          stamp['U_C'],
                                                          stamp['U_D'])

    pi_debiased, noise_arr, ston_arr = ac.construct_new_ston_cutout(stamp['I'],
                                                                    stamp['Q_A'],
                                                                    stamp['Q_B'],
                                                                    stamp['Q_C'],
                                                                    stamp['Q_D'],
                                                                    stamp['U_A'],
                                                                    stamp['U_B'],
                                                                    stamp['U_C'],
                                                                    stamp['U_D'],
                                                                    foreground_vector,
                                                                    sigma_qu)

    ston[y_min:y_max, x_min:x_max] = ston_arr
    if return_surrounding_pixels_above_threshold:
        above_thres = bool(ston[y,x]>=thres and ston[y-1,x]>=thres and ston[y,x-1]>=thres and ston[y+1,x]>=thres and ston[y,x+1]>=thres )
        return float(ston[y,x]), above_thres
    else:
        return float(ston[y,x])


RM_code_StoN("mf2", 716, 705)
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 