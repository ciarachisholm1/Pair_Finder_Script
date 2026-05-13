#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:04:48 2023

@author: ciarachisholm
"""

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import math
import time
# 
def try_path(mosaic, directory):
    from astropy.io import fits
    
    passed = None
    while passed is None:
        try: 
            hdu_listI = fits.open(directory  +mosaic+"_1420_MHz_I_image.fits")
            passed=True
        except:
            mosaic = input(mosaic +" is an invalid mosaic name entered, please enter a valid one: ")
    return mosaic


# /Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/CGPS2012/mg0_1420_MHz_I_image.fits
# /Users/ciarachisholm/Desktop/Research/CGPS2012/mg0_1420_MHz_I_image.fits
def PIimg(mosaic="ma1", plot=True, return_StoN=False, gal_coord=True, click_coord=False, create_fits=False,new_img_dir =None):

    """This function plots the polarized intensity of a CGPS mosaic and can create PI fits files. It assumes that matplotlib and 
    numpy have been imported into the script, and that astropy has been installed on the device.
    This function can print the pixel coordinates of where the cursor is clicked. 
    
    Parameters:
        mosaic (string): The name of mosaic to be shown (default: ma1)
        
        plot (boo): Whether or not to plot the function (Default = true)
        
        return_StoN (boo): Whether or not to return the signal to noise of image
        
        gal_coord (bool): Whether to display the axis in galactic coordinates. Default is True
        
        click_coord (bool): Whether to enable the pixel coordinates to print where the cursor clicks the image. Default is False
        
        create_fits (bool): whether to create new fits image of the PI. Default is False
        
        new_img_dir (str): the directory for the new fits image, default is False. 
        
    Returns:
        PI_cor: the corrected Polarized intensity of the mosaic
        
        SignToNoise (optional): the signal to noise of each pixel in the mosaic if selected
        """
    from astropy.io import fits
    from astropy.wcs import WCS
    import parameters_file as pf
    from astropy.io import fits
    
    vmax_scale = 1
    
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
    # breakpoint()
    
    if create_fits:
        if new_img_dir == None:
            new_img_dir = pf.img_dir
        
        new_img_filename = new_img_dir +mosaic+"_1420_MHz_" +"PI" +"_image.fits"
        
        
        
        PIheader = headerI.copy()
        PIheader["CTYPE4"] = "Polarised Intensity".upper()
        PIheader["NAXIS4"] = 5
        hdu = fits.PrimaryHDU(data=PI_cor, header = PIheader)
        hdu.writeto(name = new_img_filename, overwrite=True)
        
    if plot == True:
        
        
        #Plotting the image
    
        fig, ax = plt.subplots(1,1,dpi = pf.DPI, figsize=(8,8))
        # fig = plt.figure(dpi=pf.DPI, figsize=(8,8))
        # ax = plt.subplot(projection = wcs_helix)

        # plt.figure(dpi = pf.DPI, figsize=(8,8))
        # ax.set_title("The Polarised Intensity of "+mosaic.upper(), fontsize=30)
        ax.set_title(mosaic.upper()+": Polarised Intensity", fontsize=30)

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
        PLT = ax.imshow(PI_cor, vmin=pf.PI_VMIN, vmax=pf.PI_VMAX*vmax_scale,cmap="gist_heat",origin='lower',)
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)
        
        cbar = fig.colorbar(PLT,cax=cax, )
        cbar.set_label(label="Polarised Intensity (mJy/beam)", size=20)
        ticksforcbar = np.linspace( pf.PI_VMIN, pf.PI_VMAX*vmax_scale ,7)
        cbar.set_ticks(ticksforcbar.tolist())
        # cbar.ax.tick_params(labelsize=12)
        labels_for_colorbar = np.round(np.copy(ticksforcbar)*1000,1)
        cbar.ax.set_yticklabels(labels_for_colorbar.tolist())

        
        # ax.ticker.MaxNLoactor(nbins="auto")
        ax.xaxis.set_major_locator(plt.MaxNLocator("auto"))
        ax.yaxis.set_major_locator(plt.MaxNLocator("auto"))
        # overlay = ax.get_coords_overlay('icrs')
        # overlay.grid(color='white', ls='dotted')
        
        
        plt.draw()
        plt.tight_layout()
        plt.show()
    
    if return_StoN == False:
        return PI_cor
    else:
        return PI_cor,  SignToNoise


# Pi = PIimg("Mx1")



# pi,StoN = PIimg("Mx1", return_StoN=True, plot=True, gal_coord=False)
# print("StoN: ", StoN[935, 795])
# plt.figure()
# plt.imshow(StoN, origin="lower")
# # pi = PIimg("mg1")
# print("Max: ", np.nanmax(pi))
# print("ran")

def get_galactic_coor(mosaic="ma1"):
    """This function returns the axis labels for a mosaic in the CGPS.
    
    Parameters:
        mosaic (Str): the mosaic name. default is ma1
    
    Returns:
        xaxis_labels (array): The labels for the x axis. Galactic longitude in degrees
        yaxis_labels (array): The labels for the y axis. Galactic latitude in degrees"""
    
    from astropy.io import fits
    from astropy.wcs import WCS
    import parameters_file as pf
    
    
    #total intensity
    hdu_listI = fits.open(pf.img_dir  +mosaic+"_1420_MHz_I_image.fits")

    #Getting the mosaic information for the coordinates 
    headerI = hdu_listI[0].header
    
    # getting and removing unnecessary dimensions from the data
    TI_image = np.squeeze(hdu_listI[0].data)
    
    w = WCS(headerI)
    #making an array with the number of pixels in the image
    ticksx = np.linspace(0, len(TI_image[0]), len(TI_image[0]))
    ticksy = np.linspace(0, len(TI_image[:,0]), len(TI_image[0]))
    
    #Using the information from the header and the number of pixels determining 
    # what the coordinates of the image is 
    wx, wy, f, meh = w.all_pix2world(ticksx, ticksy,0,0,1)
    
    # Setting the number of ticks to be displayed on the plot
    tck = [n for n in range(0,len(TI_image[0]),1)]
    # Getting the labels of the x and y ticks
    tickx_labels = np.round(wx[tck],2)
    ticky_labels = np.round(wy[tck],2)
    
    return tickx_labels, ticky_labels
    

    

def maskblack(image,  sigma=3, black=None):
    """This function masks values in an array lower than the given minimum, 
    orignal intention to reduce noise. The function uses sigma clipped stats to 
    find the standard deviation of the image, and depending on the parameters entered
    will either set the values below a certain threshold is 0, or any value lower
    than an integer multiple of the standard deviation in the image. 
    
    Parameters:
        Image(array): the image to be masked
        sigma (int): the integer multiple to multiply the standard deviation with 
                     to set the threshold. Automatically set to 3
                    
        Black(float): the threshold value for which any value lower in the array will be set
                to display as black. Automatically set to None. 
        Returns:
            image(array): the masked image"""
    from astropy.stats import sigma_clipped_stats
    # Computing the noise reduction if the black variable is not None, and using the 
        # threshold given
    if black != None:
        # Creating an array of True and False values where the are True if they 
        #   are below threshold and False if they are above
        mask = image <= black
        
        # Setting the values that were true in the array to be 0 in our image
        image[mask] =0
        
        # Returning the image
        return image
    # Computing the noise reduction if the black variable is None, and using the 
        # mulitple of standard deviation as the threshold 
    else:
        # Calculating the mean value, median value and the standard deviation of
        # of the image given
        mean, median, sd = sigma_clipped_stats(image)
        
        # Setting the values that are below the threshold to be zero in the image 
        image[image <= sigma*sd] =0
        # Returning the image
        return image


# Pi = PIimg("mu1",True)

# # mosaic_edge_cut_out(PI_image, Mo,  plot=False,)
# PIcut, mask = mosaic_edge_cut_out(Pi, "ma1")

# mask = im1 <0.0023


# im1[mask] = 0

# PI, StN = PIimg("met2",return_StoN=True, plot=1, gal_coord=False)

# print("lobe 1: PI", PI[433, 848])
# print("lobe 1: StN", StN[433, 848])
# TI = T_Inten("met2", plot=0)
# print("lobe 1: SI", TI[433, 848])

# print("\nlobe 2: PI", PI[449, 861])
# print("lobe 2: StN", StN[449, 861])
# TI = T_Inten("met2", plot=0)
# print("lobe 2: SI", TI[449, 861])



# im2=np.copy(im1)

def T_Inten(mosaic="ma1", plot=1, click_coord=False, T=None):
    """Display the total instensity of a mosaic from the CGPS. It assumes that matplotlib and 
    numpy have been imported into the script, and that astropy has been installed on the device.
    
    Parameters:
        mosaic (string): The name of mosaic to be shown (default: ma1)
        
        plot (boo): Whether or not to plot the function (Default = true)
        
        click_coord (Boo): Whether or not to print the pixel coordinates where you click on the image. Default is False
        
    Returns:
        imI (np.array): the total intensity image of the mosaic.
        """
    from astropy.io import fits
    from astropy.wcs import WCS
    import parameters_file as pf

    vmax_scale = 0.5
  
    
    #total intensity
    hdu_listI = fits.open(pf.img_dir  +mosaic+"_1420_MHz_I_image.fits")

    #Getting the mosaic information for the coordinates 
    headerI = hdu_listI[0].header
       
    # print(headerI)
    # print(repr(headerI))
    # getting and removing unnecessary dimensions from the data
    TI_image = np.squeeze(hdu_listI[0].data)
    
    # have_problem = TI_image<0
    # no_problem = have_problem==False
    
    # should_be_zero = TI_image <1*10**(-7)
    # overlay = np.ones(shape=(TI_image.shape))
    # overlay[no_problem]=np.nan
    # TI_image[should_be_zero] =0
    # TI_image[have_problem] = np.nan
    # if np.sum(have_problem) ==0:
    #     print("it's fine")
    # else:
    #     print("Husten we have a problem")
    #     print("have_proboem: ", have_problem)
    
    
    # Adding the galactic coordinates to the image, the coordinates will 
    # not change between the files so any header can be used for this 
    w = WCS(headerI)
    #making an array with the number of pixels in the image
    ticksx = np.linspace(0, len(TI_image[0]), len(TI_image[0]))
    ticksy = np.linspace(0, len(TI_image[:,0]), len(TI_image[0]))
    
    #Using the information from the header and the number of pixels determining 
    # what the coordinates of the image is 
    wx, wy, f, meh = w.all_pix2world(ticksx, ticksy,0,0,1)
    
    # Setting the number of ticks to be displayed on the plot
    tck = [n for n in range(0,len(TI_image[0]),1)]
    # Getting the labels of the x and y ticks
    tickx_labels = np.round(wx[tck],2)
    ticky_labels = np.round(wy[tck],2)
    
    
    # plt.imshow(overlay, vmin=0,vmax=1, alpha=1, cmap="cool_r", origin="lower")
    # plt.show()

    if plot:    
        
        #Plotting the image
        fig, axs = plt.subplots(1,1, dpi = pf.DPI, figsize=(8,8))

        
        # if click_coord:
        #     def mouse_event(event):
        #         print('x: {} and y: {} in pixel coordinates'.format(np.round(event.xdata,2), np.round(event.ydata)))
            
        #     cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
        
        if T==None:
            T = mosaic.upper()+ ": Stokes I"
        
        plt.title(T, fontsize=30)
        # axs.set_title("Stokes I of a Fraternal \n Double Source", fontsize=30)
        # plt.title("The total intensity of the identical double source")
        axs.set_xticks(tck, tickx_labels)
        axs.set_yticks(tck, ticky_labels)
        
        axs.set_xlabel(r"Longitude $(\degree)$", fontsize=25)
        axs.set_ylabel (r"Latitude $(\degree)$", fontsize=25)
        
        # axs.imshow(overlay, vmin=0,vmax=1, alpha=1, cmap="cool_r", origin="lower")
        # CMAP = plt.cm.gist_heat
        # CMAP.set_bad('lime',1.)
        # PLT = axs.imshow(TI_image, vmin=pf.TI_VMIN, vmax=pf.TI_VMAX, cmap=CMAP,origin='lower')
        # PLT = axs.imshow(TI_image, vmin=pf.TI_VMIN, vmax=pf.TI_VMAX*vmax_scale, cmap="gist_heat",origin='lower')
        PLT = axs.imshow(TI_image*1000, vmin=pf.TI_VMIN, vmax=pf.TI_VMAX*vmax_scale*1000, cmap="gist_heat",origin='lower')

        # axs.imshow(overlay, vmin=0,vmax=1, alpha=1, cmap="cool_r", origin="lower")

        # plt.colorbar(cax=plt.axes([0.93, 0.11,0.02,0.76]))
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.3)
        
        cbar = fig.colorbar(PLT,cax=cax)
        # ticksforcbar = np.linspace( pf.TI_VMIN, pf.TI_VMAX*1 ,7)
        # cbar.set_ticks(ticksforcbar.tolist())
        # print("line 373 test: ", pf.TI_MIN-1)

        # labels_for_colorbar = np.round(np.copy(ticksforcbar)*1000,1)
        # cbar.ax.set_yticklabels(labels_for_colorbar.tolist())
        cbar.set_label(label="Stokes I (mJy/beam)", size=20)
        plt.tight_layout()
        
        #The following 3 lines allows the axis to update when the window is zoomed in on. 
        axs.xaxis.set_major_locator(plt.MaxNLocator("auto"))
        axs.yaxis.set_major_locator(plt.MaxNLocator("auto"))
        plt.tight_layout()
        plt.draw()
  
    
    
    return TI_image#, copyimI

Mo= "ma1"
Pi = PIimg(Mo,True, )
Si = T_Inten(Mo)
# print("SI max: ", np.nanmax(si))



def Identify_Point_Sources(Image, plot=True, vmin = 0, vmax = 0.003, threshold=1.5/2000):
    """This function identifies point sources in the mosaics, and plots them if desired. 
    It utilizes the function blob_log (which uses the laplacian of gaussian
    method) from scikit image to identify the point sources. The circles 
    plotted afterwards is also based on the example code provided on the website  
    of the package. The link to the API reference is: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_log 
    The link to the example is code is: https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html#sphx-glr-auto-examples-features-detection-plot-blob-py 
    
    Parameters:
        Image (2d array): The image with the point sources you wish to identify.
        
                          
        plot (boo):       True if you wish to point the points, set to False if you do not.
        
        vmin (float):     The minimum value you wish the colormap to have, autoset to 0.
        
        vmax (float):     The max value you wish the colormap to have, autoset tp 0.003.
        
        threshold (float):
            the threshold you in scale space (half the max of the absolute value 
            the input image) that you wish to detect. 
            Default is 0.00075 (half the CGPS PI point source threshold). 
    Returns: 
        pointsources (array):  Returns a 2D array with everyrow containing the coordinates of a source
                               and the standard deviation of the source of the form (u,v,HWHM) where 
                               u and v are pixel coordinates and HWHM is half width half max of the gaussian fit. 
         
    """
    from skimage.feature import blob_log
    import parameters_file as pf
    # Finding all the sources
    
    pointsources = blob_log(Image, max_sigma=pf.max_sigma, min_sigma=pf.min_sigma, 
                            threshold=threshold, overlap=pf.overlap_LoG, num_sigma=int(pf.num_sigmas))
    # PScircles = pointsources[:,2]*np.sqrt(2)

    
    # Creating a new list with all the radii being the HWHM of the orginal source
    #   in the produced image
    PScircles = np.copy(pointsources)
    PScircles[:,2] = PScircles[:,2]*np.sqrt(2*np.log(2)) 
    
    # Plotting the function if desired by the user. 
    if plot:
        # Creating the new plots
        fig, ax = plt.subplots(1,1, figsize=(8,8))
        ax.imshow(Image, vmin = vmin, vmax = vmax, cmap="gist_heat",origin='lower')
        # Adding the circle for each Identified "blob"
        for PS in PScircles:
            # print(PS)
            if PS[2] > pf.min_TI_radius:
                y,x,r = PS
                #Creating the circle
                circle = plt.Circle((x,y), r, color="aqua", linewidth=2, fill=False)
                #plots the circle
                ax.add_patch(circle)
            
        plt.show()
    
    return PScircles

# Mo="ma1"
# PIim1 = PIimg(mosaic ="mg5",   plot=0, gal_coord=True)
# Pi = PIimg(plot=0)


# ## Seyfret Gal stuff
# Y,X,R = Ss[:,0].astype(int), Ss[:,1].astype(int), Ss[:,2].astype(int)

# Xnot_too_small= X >= 390
# Xnot_too_big = X<= 405
# Ynot_too_small = Y>= 770
# Ynot_too_big = Y<= 810

# in_right_place = Xnot_too_big*Xnot_too_small*Ynot_too_big*Ynot_too_small
# indices = np.nonzero(in_right_place)[0]
# for i in indices:
#     print("Coordinates of the source and HWHM: ", Ss[i])

# ## KR 144 (in MX1)
# Ss = Identify_Point_Sources(PIim1[708:722,696:712], plot=True)
# print(Ss*20)
# # dx = np.abs(Ss[0,0]-Ss[1,0])
# dy = np.abs(Ss[0,1]-Ss[1,1])
# print("distance is: ", np.sqrt(dx**2+dy**2))

# SI = T_Inten("mk1")





def cm_generator(y_true, y_pred, labels, filename=None, title="", ymap=None, figsize=(10,8), fontsize_title=24,
                 fontsize_annot = 12, rotatex=False, show_fig=True, svg_dir=None):
    # From https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7 
    # I modified it to have a title, and the name of the function.
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    
    Note: this function uses the package seaborn, it must be installed for it produce a plot
          It also uses the package pandas as pd, and sklearn
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      
      y_pred:    prediction of the data, with shape (nsamples,)
      
      filename:  filename of figure file to save
      
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
                 
      title:     title of the graph, str
      
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
                 
      figsize:   the size of the figure plotted.
      
      fontsize_title: fontsize of the title, int
      
      svg_dir: where to save the svgs files to. if none entered they are not saved. 
      
    Returns:
    cm: the colormap, a 2D array.
    """
    from seaborn import heatmap
    from sklearn.metrics import confusion_matrix

    from CM_cmap import my_CM_cmap
    # rotatex=True
    
    plt.rcParams.update({'font.size': fontsize_annot})
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc1 = cm / cm_sum.astype(float) * 100
    cm_perc  = np.nan_to_num(cm_perc1, nan=222222)
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    VMAX = np.max(cm)
    
    
    if len(labels)<=3:
        empty="0"
    else:
        empty=""
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            # ptest = cm_perc[2,0]
            # print(cm_perc[2,0])
            
            if i == j:
                if p == 222222:
                   annot[i,j] ="None\n0"
                else:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = empty
             
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    # fig, ax = plt.subplots(1,1, fig_kw={"figsize":figsize})
    hm= heatmap(cm, vmin=0.0, vmax=VMAX+0.00001, annot=annot, cmap=my_CM_cmap, fmt='', ax=ax,
                      annot_kws={'size': fontsize_annot})
    # hm= heatmap(cm, vmin=0.0, vmax=VMAX+0.00001, annot=annot, cmap="rocket", fmt='', ax=ax,
    #                   annot_kws={'size': fontsize_annot})
    # hm.set_xticklabels(fontsize=fontsize_annot)
    # hm.set_yticklabels(fontsize=fontsize_annot)
    if rotatex:
        hm.set_xticklabels(hm.get_xticklabels(), rotation = 30, ha="right")
    # hm.fontsize(1)
    # plt.imshow(cm, annot=annot, )
    plt.title(title, fontsize=fontsize_title)
    plt.tight_layout()
    
    
    # plt.tight_layout()
    if filename != None:
        plt.savefig(filename)
        if svg_dir != None:
            plt.savefig(svg_dir, format="svg")
        if show_fig==False:
            plt.close()
    
    return cm

def get_AGNs_simbad(lmin, lmax, bmin, bmax, csv_dir = None):
    """This function reads in the AGNs from simbad and returns the long and lat in area specificed area. 
        Function requires numpy and pandas to run.
    Parameters:
        lmin (float): the minimum constraint on longitude 
        lmax (float): the max constraint on longitude 
        bmin (float): the minimum constraint on latitude 
        bmax (float): the max constraint on latitude
        csv_dir (str): the directory to get the AGNs from. Default is simbad sources
                        in CGPS region. Must be a csv file readable by pandas
    
    Return: 
        AGNs (2D array): array containing the coordinates in the area. each row is a source, and column 1 is the long, column 2 is the lat."""
    
    
    # if mosaic.lower() =="mg0":
    #     region = "SLE"
    # elif int(mosaic[2])>2:
    #     region = "NLE"
    # else:
    #     region = "CGPS"
    
    
    
    if csv_dir == None:
        csv_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Data Verification/SIMBAD_type_sep/all_regions_simbad.csv"

        # csv_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Data Verification/MySimbadCSVs/combined_AGNs_condensed.csv"
    
    ndf = pd.read_csv(csv_dir)
    ndf["Index"] = np.array(ndf.index.values)

    
    longs = ndf["l"].to_numpy()
    lats = ndf["b"].to_numpy()
    
    srcs_in_area = (longs<lmax)*(longs>lmin)*(lats<bmax)*(lats>bmin)
    longs_in_region = longs[srcs_in_area]
    lats_in_region= lats[srcs_in_area]
    
    AGNs = np.transpose(np.array([longs_in_region, lats_in_region]))
    shortdf = ndf[srcs_in_area]
    
    dfAGNs= shortdf.rename(columns={"identifier":"Name"})
    dfAGNs = dfAGNs[["Name", "l", "b", "Index"]].copy()
    dfAGNs["database"] = "SIMBAD"
    
    
    
    
    return AGNs, dfAGNs
# QSOs, dfQSOs = get_AGNs_simbad(150, 155.31, 0.43, 5.48)


# # dfQSOs.rename(columns={"Gaia":"Name"}, inplace=True)
# print(QSOs)
def get_AGNs_CatNorth(lmin, lmax, bmin, bmax, mosaic):
    """This function reads in the AGNs from simbad and returns the long and lat in area specificed area. 
        Function requires numpy and pandas to run.
    Parameters:
        lmin (float): the minimum constraint on longitude 
        lmax (float): the max constraint on longitude 
        bmin (float): the minimum constraint on latitude 
        bmax (float): the max constraint on latitude
        mosaic (str): the name of the mosaic being examined.
        csv_dir (str): the directory to get the AGNs from. Default is simbad sources
                        in CGPS region. Must be a csv file readable by pandas
    
    Return: 
        AGNs (2D array): array containing the coordinates in the area. each row is a source, and column 1 is the long, column 2 is the lat."""
    
    
    if mosaic.lower() =="mg0":
        region = "SLE"
    elif int(mosaic[-1])>2:
        region = "NLE"
    else:
        region = "CGPS"
    
    
    # if csv_dir == None:
    folder_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Data Verification/CatNorth/"

        # csv_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Data Verification/MySimbadCSVs/combined_AGNs_condensed.csv"
    filename = region+"_CatNorth.csv"
    ndf = pd.read_csv(folder_dir + filename)
    ndf["Index"] = np.array(ndf.index.values)

    
    longs = ndf["l"].to_numpy()
    lats = ndf["b"].to_numpy()
    
    srcs_in_area = (longs<lmax)*(longs>lmin)*(lats<bmax)*(lats>bmin)
    longs_in_region = longs[srcs_in_area]
    lats_in_region= lats[srcs_in_area]
    QSOslst = []
    for n,l in enumerate(longs_in_region):
        QSOslst.append([l, lats_in_region[n]])
    QSOs = np.array(QSOslst, dtype = float)
    
    shortdf = ndf[srcs_in_area]
    dfQSOs = shortdf[["Gaia", "l", "b", "Index"]].copy()
    dfQSOs= dfQSOs.rename(columns={"Gaia":"Name"})
    dfQSOs["database"] = "Gaia"
    
    # QSOs = np.column_stack((longs_in_region, lats_in_region))
    # QSOs = np.transpose(np.array([longs_in_region, lats_in_region]))
    
    return QSOs, dfQSOs
# QSOs, dfQSOs = get_AGNs_CatNorth(150, 155.31, 0.43, 5.48, "mst2")


# # dfQSOs.rename(columns={"Gaia":"Name"}, inplace=True)
# print(QSOs)


def get_AGNs_VLA_DRAGNs(lmin, lmax, bmin, bmax, csv_dir = None):
    """This function reads in the AGNs from simbad and returns the long and lat in area specificed area. 
        Function requires numpy and pandas to run.
    Parameters:
        lmin (float): the minimum constraint on longitude 
        lmax (float): the max constraint on longitude 
        bmin (float): the minimum constraint on latitude 
        bmax (float): the max constraint on latitude
        csv_dir (str): the directory to get the AGNs from. Default is simbad sources
                        in CGPS region. Must be a csv file readable by pandas
    
    Return: 
        AGNs (2D array): array containing the coordinates in the area. each row is a source, and column 1 is the long, column 2 is the lat."""
    
    
    # if mosaic.lower() =="mg0":
    #     region = "SLE"
    # elif int(mosaic[2])>2:
    #     region = "NLE"
    # else:
    #     region = "CGPS"
    
    
    
    if csv_dir == None:
        csv_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Data Verification/VLA_DRAGNs/VLA_DRAGNs_in_CGPS.csv"

        # csv_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Data Verification/MySimbadCSVs/combined_AGNs_condensed.csv"
    
    ndf = pd.read_csv(csv_dir)
    
    ndf["Index"] = np.array(ndf.index.values)
    
    longs = ndf["l_lobe"].to_numpy()
    lats = ndf["b_lobe"].to_numpy()
    
    srcs_in_area = (longs<lmax)*(longs>lmin)*(lats<bmax)*(lats>bmin)
    longs_in_region = longs[srcs_in_area]
    lats_in_region= lats[srcs_in_area]
    
    DRAGNs = np.transpose(np.array([longs_in_region, lats_in_region]))
    shortdf = ndf[srcs_in_area]
    dfDRAGNs = shortdf[["Name", "l_lobe", "b_lobe", "Index"]].copy()
    dfDRAGNs = dfDRAGNs.rename(columns={"l_lobe":"l", "b_lobe":"b"})
    dfDRAGNs["database"] = "VLA_DRAGNs"
    
    
    return DRAGNs, dfDRAGNs

def get_GLADE_Gs(lmin, lmax, bmin, bmax, csv_dir = None):
    """This function reads in the AGNs from simbad and returns the long and lat in area specificed area. 
        Function requires numpy and pandas to run.
    Parameters:
        lmin (float): the minimum constraint on longitude 
        lmax (float): the max constraint on longitude 
        bmin (float): the minimum constraint on latitude 
        bmax (float): the max constraint on latitude
        csv_dir (str): the directory to get the AGNs from. Default is simbad sources
                       in CGPS region. Must be a csv file readable by pandas
    
    Return: 
        AGNs (2D array): array containing the coordinates in the area. each row is a source, and column 1 is the long, column 2 is the lat."""
    
    
    
    if csv_dir == None:
        csv_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Data Verification/GLADE/GLADE_in_CGPS.csv"

        # csv_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Data Verification/MySimbadCSVs/combined_AGNs_condensed.csv"
    
    ndf = pd.read_csv(csv_dir, index_col="n")
    
    
    
    longs = ndf["l"].to_numpy()
    lats = ndf["b"].to_numpy()
    ndf["Index"] = np.array(ndf.index.values)
    srcs_in_area = (longs<lmax)*(longs>lmin)*(lats<bmax)*(lats>bmin)
    longs_in_region = longs[srcs_in_area]
    lats_in_region= lats[srcs_in_area]
    
    Gs = np.transpose(np.array([longs_in_region, lats_in_region]))
    shortdf = ndf[srcs_in_area]
    dfGs = shortdf[["Primary Catalogue Number", "l", "b","Index"]].copy()
    dfGs = dfGs.rename(columns={"Primary Catalogue Number":"Name"})
    dfGs["database"] = "GLADE"
    
    
    return Gs, dfGs


def get_MORX(lmin, lmax, bmin, bmax, csv_dir = None):
    """This function reads in the double sources from MORX in the CGPS greater with
        a separation greater than 0.75 arcminutes and returns the long and lat in 
        area specificed area. 
        Function requires numpy and pandas to run.
    Parameters:
        lmin (float): the minimum constraint on longitude 
        lmax (float): the max constraint on longitude 
        bmin (float): the minimum constraint on latitude 
        bmax (float): the max constraint on latitude
        csv_dir (str): the directory to get the AGNs from. Default is simbad sources
                       in CGPS region. Must be a csv file readable by pandas
    
    Return: 
        AGNs (2D array): array containing the coordinates in the area. each row is a source, and column 1 is the long, column 2 is the lat."""
    
    
    
    if csv_dir == None:
        # csv_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Data Verification/MORX/MORX_reduced_col_in_CGPS_offset_larger_than_0.75arcmin.csv"
        csv_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Data Verification/MORX/MORX_reduced_col_in_CGPS.csv"

        # csv_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Data Verification/MySimbadCSVs/combined_AGNs_condensed.csv"
    
    ndf = pd.read_csv(csv_dir, index_col="n")
    
    
    
    longs = ndf["l"].to_numpy()
    lats = ndf["b"].to_numpy()
    ndf["Index"] = np.array(ndf.index.values)
    srcs_in_area = (longs<lmax)*(longs>lmin)*(lats<bmax)*(lats>bmin)
    longs_in_region = longs[srcs_in_area]
    lats_in_region= lats[srcs_in_area]
    
    Gs = np.transpose(np.array([longs_in_region, lats_in_region]))
    shortdf = ndf[srcs_in_area]
    dfGs = shortdf[["Name", "l", "b","Index"]].copy()
    # dfGs = dfGs.rename(columns={"Primary Catalogue Number":"Name"})
    dfGs["database"] = "MORX"
    
    
    return Gs, dfGs

def get_TayCat(lmin, lmax, bmin, bmax, csv_dir = None):
    """This function reads in the double sources from MORX in the CGPS greater with
        a separation greater than 0.75 arcminutes and returns the long and lat in 
        area specificed area. 
        Function requires numpy and pandas to run.
    Parameters:
        lmin (float): the minimum constraint on longitude 
        lmax (float): the max constraint on longitude 
        bmin (float): the minimum constraint on latitude 
        bmax (float): the max constraint on latitude
        csv_dir (str): the directory to get the AGNs from. Default is simbad sources
                       in CGPS region. Must be a csv file readable by pandas
    
    Return: 
        AGNs (2D array): array containing the coordinates in the area. each row is a source, and column 1 is the long, column 2 is the lat."""
    
    
    
    if csv_dir == None:
        # csv_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Data Verification/MORX/MORX_reduced_col_in_CGPS_offset_larger_than_0.75arcmin.csv"
        csv_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Data Verification/Taylor2017/Tay_cat.csv"

        # csv_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Data Verification/MySimbadCSVs/combined_AGNs_condensed.csv"
    
    ndf = pd.read_csv(csv_dir)
    
    
    
    longs = ndf["l"].to_numpy()
    lats = ndf["b"].to_numpy()
    ndf["Index"] = np.array(ndf.index.values)
    srcs_in_area = (longs<lmax)*(longs>lmin)*(lats<bmax)*(lats>bmin)
    longs_in_region = longs[srcs_in_area]
    lats_in_region= lats[srcs_in_area]
    
    Ts = np.transpose(np.array([longs_in_region, lats_in_region]))
    shortdf = ndf[srcs_in_area]
    dfTs = shortdf[["CGPS", "l", "b","Index"]].copy()
    dfTs = dfTs.rename(columns={"CGPS":"Name"})
    dfTs["database"] = "TayCat"
    
    
    return Ts, dfTs

def get_AGNs_VLA_large_rG(lmin, lmax, bmin, bmax, csv_dir = None):
    """This function reads in the AGNs from simbad and returns the long and lat in area specificed area. 
        Function requires numpy and pandas to run.
    Parameters:
        lmin (float): the minimum constraint on longitude 
        lmax (float): the max constraint on longitude 
        bmin (float): the minimum constraint on latitude 
        bmax (float): the max constraint on latitude
        csv_dir (str): the directory to get the AGNs from. Default is simbad sources
                        in CGPS region. Must be a csv file readable by pandas
    
    Return: 
        AGNs (2D array): array containing the coordinates in the area. each row is a source, and column 1 is the long, column 2 is the lat."""
    
    
    # if mosaic.lower() =="mg0":
    #     region = "SLE"
    # elif int(mosaic[2])>2:
    #     region = "NLE"
    # else:
    #     region = "CGPS"
    
    
    
    if csv_dir == None:
        csv_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Data Verification/VLA_large_rG/VLA_large_rG_in_CGPS.csv"

        # csv_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara\'s\ Research\ Cubby/Data\ Verification/VLA_large_rG/VLA_large_rG_in_CGPS.csv"
    
    ndf = pd.read_csv(csv_dir)
    
    ndf["Index"] = np.array(ndf.index.values)

    
    longs = ndf["l"].to_numpy()
    lats = ndf["b"].to_numpy()
    
    srcs_in_area = (longs<lmax)*(longs>lmin)*(lats<bmax)*(lats>bmin)
    longs_in_region = longs[srcs_in_area]
    lats_in_region= lats[srcs_in_area]
    
    large_rG = np.transpose(np.array([longs_in_region, lats_in_region]))
    shortdf = ndf[srcs_in_area]
    dflarge_rG = shortdf[["Name", "l", "b","Index"]].copy()
   
    dflarge_rG["database"] = "VLA_large_rG"
    
    
    return large_rG, dflarge_rG
# QSOs, dfQSOs = get_AGNs_VLA_DRAGNs(150, 155.31, 0.43, 5.48)


# # dfQSOs.rename(columns={"Gaia":"Name"}, inplace=True)
# print(dfQSOs.columns)
def AGN_associated(df, center_long, center_lat, distance_arcmin):
    """This function takes in a dataset of AGNs and whether there is a 
    source associated with the twin pair. It is associated if there is a
    source within a circular region centered at the center of the twins 
    and with a radius of length of half the distance between twins. If 
    there is more than 1 close enough it uses the closest source to the 
    center.
    
    Parameters:
        
        df (dataframe): The dataframe containing the names, longitudes,
            latitudes and the survey from where the source originates  
            about the AGNs. 
        
        center_long(float): the longtitude to center the circular region 
        
        center_lat(float): the latitude to center the circular region 
        
        distance_arcmin: the distance between the twin sources in arcminutes.
        
    Returns:
        associated (bool): whether there is an AGN associated, if not returns
            False and all other variables return None. 
            
        offset (float): the distance between the AGN and the center of the 
        region. 
        
        closeAGN_long (float): the longitude of the AGN
        
        closeAGN_lat (float): the latitude of the AGN
    """
    if len(df) ==0:
        associated=False
        offset = None
        name, closeAGN_long, closeAGN_lat, survey = None, None, None, None
    
    
    # AGN_long, AGN_lat = AGNs[:,0], AGNs[:,1]
    AGN_long, AGN_lat = df["l"].to_numpy(), df["b"].to_numpy()
    
    
    long_offset, lat_offset = AGN_long- center_long, AGN_lat - center_lat
    distance_btw_center_and_AGN = np.sqrt(long_offset**2+lat_offset**2)*60# factor of 60 to convert into arcmins instead of degrees
   
    # print("l881) min distance", distance_btw_center_and_AGN)
    
    if np.sum(distance_btw_center_and_AGN<=distance_arcmin/2+(1/6))==0: # the 1/6 allows for the radius to go out to the edge of 
        #  last pixel in case the AGN is like a couple arcseconds outside of the range
        associated=False
        offset = None
        name, closeAGN_long, closeAGN_lat, survey, Index = None, None, None, None, None
    else:
        n = np.nanargmin(distance_btw_center_and_AGN)
        associated=True

        offset = distance_btw_center_and_AGN[n]
        
        name = df["Name"].tolist()[n]
        survey = df["database"].tolist()[n]
        Index = df["Index"].tolist()[n]
        
        closeAGN_long, closeAGN_lat = AGN_long[n], AGN_lat[n]
        # if VLA:
        #     closeAGN_long, closeAGN_lat = df["l_lobe"].tolist()[n], df["b_lobe"].tolist()[n]
        # else:
        #     closeAGN_long, closeAGN_lat = df["l"].tolist()[n], df["b"].tolist()[n]

    return associated, offset, name, closeAGN_long, closeAGN_lat, survey, Index

# print(AGN_associated(pd.DataFrame([], columns = ["Name", "l", "b", "database"]), 122, 1, 2))


def pixel_to_arcseconds(iterable):
    """ This function converts pixel distances to arcseconds. This is done using 
        the knowledge that 1 pixel in the CGPS has a width of 0.004988 degrees, 
        or 17.9568".
        Note: this function requires numpy to run. 
    Parameters:
        
            iterable: A array-like object containing pixel distances to be converted
                      to arcseconds. 
        return: 
            arcsecond_iterable(narray): returns the list in arcseconds."""
            
    iterable_array= np.array(iterable)
    return iterable_array*17.9568



def mosaic_edge_cut_out(PI_image, Mo,  plot=False,):
    """This function returns an array the size of the input images of true values
    where the weight maps for the corresponding mosaic are above a certain threshold,
    and False values where the image is a NaN or below threshold.
    
    The intent is to use the function the return to create a mask for the PI 
    images where object detection will be observed. 
    
    Note: the following must have been previously run for the function to work:
        import matplotlib as plt
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS
 
  
    
    Key Parameters:
        PI_image (2D array): The polarised intensity image of the mosaic. 
        
        Mo (str): the mosaic in questions.
        
        plot: Whether to plot the new weight map. 
        
        
    return:
        mask (2D array):An array of boolean values the size of the mosaic. 
                        Values are True if the weight was below the given threshold,
                        and False if it was above or a NaN."""
    
    from astropy.io import fits
    from astropy.wcs import WCS
    import parameters_file as pf
    
    vmin = 0
    vmax = 1
    #Loading the weights fits file                    
    hdu_listA = fits.open(pf.img_dir  +Mo+"_1420_MHz_POL_A_wght.fits")
    hdu_listB = fits.open(pf.img_dir  +Mo+"_1420_MHz_POL_B_wght.fits")
    hdu_listC = fits.open(pf.img_dir  +Mo+"_1420_MHz_POL_C_wght.fits")
    hdu_listD = fits.open(pf.img_dir  +Mo+"_1420_MHz_POL_D_wght.fits")
    
    
    
    
       
    # getting and removing unnecessary dimensions from the data
    imA = np.squeeze(hdu_listA[0].data)
    imB = np.squeeze(hdu_listB[0].data)
    imC = np.squeeze(hdu_listC[0].data)
    imD = np.squeeze(hdu_listD[0].data)
    
    # Finding the average weight. 
    ImW_ave = (imA + imB+imC+imD)/4
    
    # Finding where there were NaNs in the fits files for the mosaic
    OGNaNs = np.isnan(PI_image)
    
    # Creating an array of booleans where the values are true if the corresponding 
    #   pixels are below threshold
    below_threshold = ImW_ave<= pf.mosaic_edge_weight_threshold
    
    # Setting the values of the pixels that are NaNs to be false. 
    below_threshold[OGNaNs] = False
    
   
    
    if plot:
        #Getting the mosaic information for the coordinates 
        headerA = hdu_listA[0].header
        
        
        # Adding the galactic coordinates to the image, the coordinates will 
        # not change between the files so any header can be used for this 
        w = WCS(headerA)
        #making an array with the number of pixels in the image
        ticksx = np.linspace(0, len(imA[0]), len(imA[0]))
        ticksy = np.linspace(0, len(imA[:,0]), len(imA[0]))
        
        #Using the information from the header and the number of pixels determining 
        # what the coordinates of the image is 
        wx, wy, f, meh = w.all_pix2world(ticksx, ticksy,0,0,1)
        
        # Setting the number of ticks to be displayed on the plot
        tck = [n for n in range(0,1024,pf.num_of_pixels_btw_ticks)]
        # Getting the labels of the x and y ticks
        tickx_labels = np.round(wx[tck],2)
        ticky_labels = np.round(wy[tck],2)
        
        #Plotting the image
        fig, axs = plt.subplots(1,1)
        # plt.title("The Total Intensity of mosaic "+mosaic, fontsize=20)
        axs.set_title("Weights of mosaic " + Mo.upper(), fontsize=30)
        # plt.title("The total intensity of the identical double source")
        axs.set_xticks(tck, tickx_labels)
        axs.set_yticks(tck, ticky_labels)
        axs.set_xlabel(r"Longitude $(^\circ)$", fontsize=25)
        axs.set_ylabel (r"Latitude $(^\circ)$", fontsize=25)
        # PLT = axs.imshow(imA, vmin=-VMIN, vmax=VMAX, cmap="gist_heat",origin='lower')
        PLT = axs.imshow(ImW_ave, cmap="gist_heat",origin='lower')
        
        
        # Creating the mask for the plot
        mask = np.ones(shape=PI_image.shape)
        
        mask[np.invert(below_threshold)] = np.nan
        axs.imshow(mask, alpha=1,origin='lower', vmax=1, vmin=0)
        # plt.colorbar(cax=plt.axes([0.93, 0.11,0.02,0.76]))
        
        cbar = fig.colorbar(PLT,)
        ticksforcbar = np.linspace(vmin, vmax ,6)
        cbar.set_ticks(ticksforcbar.tolist())
        plt.tight_layout()
        
        
    return below_threshold


# the following has most artifacts masked out, more than is stated in the thesis. 
# def cut_out_for_mosaic(PIim1,Mo,  overlap=True):
#     """ This function produces a mask cutouts for mosaics that will remove any previously
#     identified artifacts, and removes the overlap region if set to. Anywhere that is masked
#     will have a NaN value. 
    
#     Parameters:
#         PIim1 (2D array): The polarised intensity image produced for the given array.
        
#         Mo (String): The mosaic of the polarised intensity image.
        
#         overlap (boo): Whether to give an overlap mask or not (Default to true)
        
#     Returns: 
#     if overlap is True: 
        
#         PIcutout (2D array): The polarised intensity image with the cuts to the 
#                              map.
                             
#         Overlapcutout (2D array): An array of values that are True if the corresponding
#                                   pixel if was masked out due to overlap. 
        
#         maskforoverlap: Returns a mask for the overlap region where everyvalue 
#                         in the array is zero (if the area is masked) or a NaN. 
        
#         maskforplot(2D array): Returns a mask for the edges region where everyvalue 
#                         in the array is 1 (if the area is masked) or a NaN.
#     if overlap is False: 
#         PIcutout (2D array): The polarised intensity image with the cuts to the 
#                              map.
                             
#         maskforplot(2D array): Returns a mask for the edges region where everyvalue 
#                         in the array is 1 (if the area is masked) or a NaN.
#             """
    
#     import parameters_file as pf
    
    
    
#     # Creating an array to store where the orignal NaNs in the mosaic occured
#     #   (so the places with no field observations can be set to white in the plot)
#     OGNaNs = np.isnan(PIim1)

#     # copying the orignal array
#     PIcutout = np.copy(PIim1)
    
#     # Getting where the edges of the fields are, the edges in the fields have do
#     #   not have the same sensitivity as the centers and the signal to noise ratio
#     #   is lower in this regions. The weight of the fields is proportional to the 
#     #   signal to noise for the field, and the edges do not have any overlapping 
#     #   regions therefore they have the lowest weights. Any pixel below the given 
#     #   weight will be set to True, where the rest are set to False. 
#     below_threshold = mosaic_edge_cut_out(PI_image=PIim1, Mo=Mo, plot=False)
#     PIcutout[below_threshold] = np.nan



    
#     # Checking if the Mosaic is MF1, there are sections of that mosaic that need to be removed
#     # from the area we want to check. Specifically CAS A and another cut out region that needs
#     # to be expanded due to leakage. 
#     if Mo=="mf1":
        
        
#         # Removing CAS A and it's noisy surrounding regions from the detecting image
#         PIcutout[90:500, 0:200] = np.nan

#         PIcutout[440:565, 360:490] = np.nan # SNR G109.0-1.0
        
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[(-1*pf.mosaic_overlap_width):,:] = True
#             overlapcutout[:,:pf.mosaic_overlap_width] = True
#             overlapcutout[:,(-1*pf.mosaic_overlap_width):] = True
            
    
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
#     # Checking if the Mosaic is ME1, there are sections of that mosaic that need to be removed
#     # from the area we want to check. Specifically CAS A and the bottom left corner is missing,
#     # and the surrounding region has more noise than the rest of the mosaic. 
#     elif Mo=='me1':
        
#         # Cutting out the sections with leakage, or have curved edges leading to false detections
        
#         PIcutout[:600, 400:] = np.nan
#         PIcutout[850:, 720:790] = np.nan
        
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
#             overlapcutout[:,0:pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
    
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
#     elif Mo=='mn1':
        
#         # Cutting out the sections with leakage, HII regions, or have curved edges leading to false detections
        
#         PIcutout[800:840, 300:350] = np.nan
#         PIcutout[645:705, 450:500] = np.nan
#         PIcutout[720:810, 770:830] = np.nan
        
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
#             overlapcutout[:,0:pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
    
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
    
#     # Checking if the Mosaic is MO2. One of the observations 
#     elif Mo=="mo2":
        
#         # The odd source at the top cut out. 
#         PIcutout[-350:, 300:850] = np.nan
#         PIcutout[120:170, 850:900] = np.nan
#         PIcutout[200:325, 150:300] = np.nan
        
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[:pf.mosaic_overlap_width,:] = True
#             overlapcutout[:,0:pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
    
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
#     elif Mo=="mej2":
        
#         # The odd source at the top cut out. 
#         PIcutout[508:518, 612:623] = np.nan
        
        
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[:pf.mosaic_overlap_width,:] = True
#             overlapcutout[:,0:pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
    
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
            
#     # Checking if the mosaic is one on the edge of the mosaic. 
#     elif Mo=='meq2' or Mo=='meq1' or Mo=='mel1' or Mo=='mel2':
        
        
#         if overlap:
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             if Mo[-1]=="1":
#                 overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
#             else:
#                 overlapcutout[:pf.mosaic_overlap_width,:] = True 
#             if Mo[2]=="q":
#                 overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
#             else:
#                 overlapcutout[:,0:pf.mosaic_overlap_width] = True
#             ### Overlapping regions in the mosaics

#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan

#      # Checking if the mosaic is a side of the extension  into the halo of the CGPS 
#     elif Mo=="me3" or Mo=="me4" or Mo=="me5" or Mo=="mh3" or Mo=='mh4' or Mo=='mh5':

         
         
#          if overlap:
#              # Creating an array to show the overlap region in the mosaic. The true values 
#              #   will be set to NaNs later in the code to show the mask overlap region
#              overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
             
#              if Mo[-1]=="5":
#                  print(Mo[2])
#                  overlapcutout[:pf.mosaic_overlap_width,:] = True
#              else:
#                  overlapcutout[:pf.mosaic_overlap_width,:] = True 
#                  overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
             
#              if Mo[1]=="e":
#                  overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
#              else:
#                  overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
#              ### Overlapping regions in the mosaics
    
#              # Cutting out the overlap regions from the polarized intensity image that 
#              #   will be used to detect the point sources within it. 
#              PIcutout[overlapcutout] = np.nan
#     elif Mo=="md2" or Mo=='mh2':
        
    
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions that occur in both mosaics
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
#             overlapcutout[:,:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:1*pf.mosaic_overlap_width,:] = True
            
#             # Cutting out the overlap regions that only occur in each mosaic. 
#             if Mo=="mh2":
#                 overlapcutout[-112:,0:694] = True
                
#             else:
#                 overlapcutout[-112:,-554:] = True
                
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
#     elif Mo=="my2":
#         # Cutting out the leakage from the source in the mosaic
#         PIcutout[100:300, 200:450] = np.nan
#         PIcutout[992:938, 608:623] = np.nan
#         PIcutout[515:540, 895:915] = np.nan
        
        
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[:1*pf.mosaic_overlap_width,:] = True
#             overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
#     elif Mo=="my1":
#         # Cutting out the leakage from the source in the mosaic
#         PIcutout[-150:-100, 250:400] = np.nan
        
        
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
#             overlapcutout[:,:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan

#     elif Mo=="mk1":
#         # Cutting out the leakage from the source in the mosaic
#         PIcutout[475:800, 150:500] = np.nan
        
        
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
#             overlapcutout[:,:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
#     elif Mo=="mq2":
#         # Cutting out the leakage from what I assume is a gas cloud
#         PIcutout[450:550, 250:675] = np.nan
#         PIcutout[250:450, 375:525]=np.nan
#         PIcutout[250:300, 690:735]=np.nan
#         PIcutout[195:270, 160:300]=np.nan
        
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[:1*pf.mosaic_overlap_width,:] = True
#             overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
            
            
#     elif Mo=="mer1":
#         PIcutout[710:730,627:670] = np.nan
#         PIcutout[550:600,325:400] = np.nan
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
#             overlapcutout[:,:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
#     elif Mo=="mer2":
        
#         PIcutout[400:595,360:540] = np.nan
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[:1*pf.mosaic_overlap_width,:] = True
#             overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
            
#     elif Mo=="mes2":
        
#         PIcutout[672:750,282:385] = np.nan
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[:1*pf.mosaic_overlap_width,:] = True
#             overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
            
#     elif Mo=="mez2":
        
#         PIcutout[920:950,775:805] = np.nan
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[:1*pf.mosaic_overlap_width,:] = True
#             overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
            
#     elif Mo=="mey1":
#         PIcutout[684:753,561:612] = np.nan
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
#             overlapcutout[:,:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
#     elif Mo=="mv1":
#         PIcutout[483:522,657:705] = np.nan
#         PIcutout[210:270,390:440] = np.nan
#         PIcutout[660:690,615:640] = np.nan
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
#             overlapcutout[:,:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
            
#     elif Mo=="ml2":
        
#         PIcutout[153:195,132:180] = np.nan
#         PIcutout[690:,264:732] = np.nan
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[:1*pf.mosaic_overlap_width,:] = True
#             overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
#     elif Mo=="mn2":
        
#         PIcutout[900:,132:350] = np.nan
#         PIcutout[150:200,770:830] = np.nan
        
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[:1*pf.mosaic_overlap_width,:] = True
#             overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
#     elif Mo=="mo1":
#         PIcutout[759:825,663:732] = np.nan
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
#             overlapcutout[:,:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
#     elif Mo=="mr2":
        
#         PIcutout[117:186,285:350] = np.nan
        
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[:1*pf.mosaic_overlap_width,:] = True
#             overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
            
#     elif Mo=="met2":
        
#         PIcutout[400:475,875:950] = np.nan
        
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[:1*pf.mosaic_overlap_width,:] = True
#             overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
#     elif Mo=="mew2":
        
#         # PIcutout[400:475,825:950] = np.nan
        
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[:1*pf.mosaic_overlap_width,:] = True
#             overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
#             overlapcutout[400:,-1*pf.mosaic_overlap_width:] = True
#             overlapcutout[350:400,-1*pf.mosaic_overlap_width +20:] = True
#             overlapcutout[:350,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
#     elif Mo=="ma2":
        
#         PIcutout[500:550,100:150] = np.nan
        
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[:1*pf.mosaic_overlap_width,:] = True
#             overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
#     elif Mo=="mc2":
        
#         PIcutout[150:250,600:700] = np.nan
#         # 
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[:1*pf.mosaic_overlap_width,:] = True
#             overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
#     elif Mo=="mf2":
        
#         PIcutout[830:880,100:170] = np.nan
#         PIcutout[728:760,694:735] = np.nan
#         # 
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[:1*pf.mosaic_overlap_width,:] = True
#             overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
#             # overlapcutout[555:590,0:1*pf.mosaic_overlap_width-15] = True
#             # overlapcutout[590:,0:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
            
    
#     elif Mo=="mej1":

#         PIcutout[700:760,330:410] = np.nan
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
#             overlapcutout[:,:1*pf.mosaic_overlap_width] = True
#             overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
    
#     elif Mo=="mx1":

#         # PIcutout[700:760,330:410] = np.nan
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
#             overlapcutout[:,:1*pf.mosaic_overlap_width] = True
#             # overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
#             overlapcutout[:440,-1*pf.mosaic_overlap_width:] = True
#             overlapcutout[440:495,-1*pf.mosaic_overlap_width +30:] = True
#             overlapcutout[495:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
    
#     elif Mo=="mg0":
#         if overlap:
#             ### Overlapping regions in the mosaics
            
#             # Creating an array to show the overlap region in the mosaic. The true values 
#             #   will be set to NaNs later in the code to show the mask overlap region
#             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
#             # Setting the overlap regions to be false so later they will show up when
#             #   the mask is plotted. 
#             overlapcutout[-379:,:] = True
#             # overlapcutout[:,:1*pf.mosaic_overlap_width] = True
#             # overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
#             # Cutting out the overlap regions from the polarized intensity image that 
#             #   will be used to detect the point sources within it. 
#             PIcutout[overlapcutout] = np.nan
    
#     else:
        
#         # If there are NaN values in the array, see if in the botton right corner
#         #   (indicating the bottom of the mosaic should be cut out) or the top right
#         #   corner (indicating the top should be cut out). 
#         if np.isnan(PIim1).any():  
#             if overlap:
#                 if np.isnan(PIim1[0,0]):
#                     ### Overlapping regions in the mosaics
                    
#                     # Creating an array to show the overlap region in the mosaic. The true values 
#                     #   will be set to NaNs later in the code to show the mask overlap region
#                     overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
#                     # Setting the overlap regions to be false so later they will show up when
#                     #   the mask is plotted. 
#                     overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
#                     overlapcutout[:,:1*pf.mosaic_overlap_width] = True
#                     overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
                    
                   
    
#                     # Cutting out the overlap regions from the polarized intensity image that 
#                     #   will be used to detect the point sources within it. 
#                     PIcutout[overlapcutout] = np.nan
#                 elif np.isnan(PIim1[-1,0]):
#                     ### Overlapping regions in the mosaics
                    
#                     # Creating an array to show the overlap region in the mosaic. The true values 
#                     #   will be set to NaNs later in the code to show the mask overlap region
#                     overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
#                     # Setting the overlap regions to be false so later they will show up when
#                     #   the mask is plotted. 
#                     overlapcutout[:1*pf.mosaic_overlap_width,:] = True
#                     overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
#                     overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
    
                   
#                     # Cutting out the overlap regions from the polarized intensity image that 
#                     #   will be used to detect the point sources within it. 
#                     PIcutout[overlapcutout] = np.nan 
                    
#                 else:
#                     print("There are NaNs somewhere in mosaic "+Mo.upper())
#                     # Creating a copy array so that they all have the same name 
#                     PIcutout= np.copy(PIim1)
            
        
        
        
        
        
        
        
#         # Indicating no NaNs were found in the image. This will only apply to the halo mosaics
#         else:
#             # Creating a copy array so that they all have the same name 
#             PIcutout =np.copy(PIim1)
#             print("No cut outs were made to mosaic " + Mo.upper())
#             if overlap:
#                 ### Overlapping regions in the mosaics
                
#                 # Creating an array to show the overlap region in the mosaic. The true values 
#                 #   will be set to NaNs later in the code to show the mask overlap region
#                 overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
#                 # Setting the overlap regions to be false so later they will show up when
#                 #   the mask is plotted. 
#                 overlapcutout[:1*pf.mosaic_overlap_width,:] = True
#                 if Mo[-1]!='5':
#                     overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
#                 overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
#                 overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
                
        
#                 # Cutting out the overlap regions from the polarized intensity image that 
#                 #   will be used to detect the point sources within it. 
#                 PIcutout[overlapcutout] = np.nan
                
                
                
#     # Creating an array of values where indicating where there is a number vs a NaN
#     maskrev = np.isfinite(PIcutout)
#     # Excluding the orignal NaN values from the mask (so we can clearly see
#     #    where there is no data)
#     maskrev[OGNaNs] = True
#     # Creating an initial array that will show be the masked area of the mosaic
#     maskforplot = np.ones(shape=(PIim1.shape))
#     # Setting the areas where there was a number in the PI to not be part of the mask. 
#     maskforplot[maskrev] = np.nan
    
#     # Creating an array for the overlap mask to be added to plots, then 
#     #    using the previously calculated overlap region (mosaic dependant)
#     #    to create the values in the plot. NaNs will not appear in the mask, 
#     #    but the zeros will.
#     if overlap:
#         maskforoverlap = np.zeros(shape=PIim1.shape)
#         maskforoverlap[np.invert(overlapcutout)] = np.nan
    
#     # Creating an array of values where indicating where there is a number vs a NaN
#     maskrev = np.isfinite(PIcutout)
#     # Excluding the orignal NaN values from the mask (so we can clearly see
#     #    where there is no data)
#     maskrev[OGNaNs] = True
#     # Cutting out the overlap region so that they don't overlap (important
#     #    for matching colormaps)
#     if overlap:
#         maskrev[overlapcutout] = True
#     # Creating an initial array that will show be the masked area of the mosaic
#     maskforplot = np.ones(shape=(PIim1.shape))
#     # Setting the areas where there was a number in the PI to not be part of the mask. 
#     maskforplot[maskrev] = np.nan
    
#     # Only returning the overlap cut out if set to. 
#     if overlap:
#         return PIcutout, overlapcutout, maskforoverlap, maskforplot
#     else: 
#         return PIcutout, maskforplot
  




# the following only has large objects masked out
def cut_out_for_mosaic(PIim1,Mo,  overlap=True):
    """ This function produces a mask cutouts for mosaics that will remove any previously
    identified artifacts, and removes the overlap region if set to. Anywhere that is masked
    will have a NaN value. 
    
    Parameters:
        PIim1 (2D array): The polarised intensity image produced for the given array.
        
        Mo (String): The mosaic of the polarised intensity image.
        
        overlap (boo): Whether to give an overlap mask or not (Default to true)
        
    Returns: 
    if overlap is True: 
        
        PIcutout (2D array): The polarised intensity image with the cuts to the 
                             map.
                             
        Overlapcutout (2D array): An array of values that are True if the corresponding
                                  pixel if was masked out due to overlap. 
        
        maskforoverlap: Returns a mask for the overlap region where everyvalue 
                        in the array is zero (if the area is masked) or a NaN. 
        
        maskforplot(2D array): Returns a mask for the edges region where everyvalue 
                        in the array is 1 (if the area is masked) or a NaN.
    if overlap is False: 
        PIcutout (2D array): The polarised intensity image with the cuts to the 
                             map.
                             
        maskforplot(2D array): Returns a mask for the edges region where everyvalue 
                        in the array is 1 (if the area is masked) or a NaN.
            """
    
    import parameters_file as pf
    
    
    
    # Creating an array to store where the orignal NaNs in the mosaic occured
    #   (so the places with no field observations can be set to white in the plot)
    OGNaNs = np.isnan(PIim1)

    # copying the orignal array
    PIcutout = np.copy(PIim1)
    
    # Getting where the edges of the fields are, the edges in the fields have do
    #   not have the same sensitivity as the centers and the signal to noise ratio
    #   is lower in this regions. The weight of the fields is proportional to the 
    #   signal to noise for the field, and the edges do not have any overlapping 
    #   regions therefore they have the lowest weights. Any pixel below the given 
    #   weight will be set to True, where the rest are set to False. 
    below_threshold = mosaic_edge_cut_out(PI_image=PIim1, Mo=Mo, plot=False)
    PIcutout[below_threshold] = np.nan



    
    # Checking if the Mosaic is MF1, there are sections of that mosaic that need to be removed
    # from the area we want to check. Specifically CAS A and another cut out region that needs
    # to be expanded due to leakage. 
    if Mo=="mf1":
        
        
        # Removing CAS A and it's noisy surrounding regions from the detecting image
        PIcutout[90:500, 0:200] = np.nan

        PIcutout[440:565, 360:490] = np.nan # SNR G109.0-1.0
        
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[(-1*pf.mosaic_overlap_width):,:] = True
            overlapcutout[:,:pf.mosaic_overlap_width] = True
            overlapcutout[:,(-1*pf.mosaic_overlap_width):] = True
            
    
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    # Checking if the Mosaic is ME1, there are sections of that mosaic that need to be removed
    # from the area we want to check. Specifically CAS A and the bottom left corner is missing,
    # and the surrounding region has more noise than the rest of the mosaic. 
    elif Mo=='me1':
        
        # Cutting out the sections with leakage, or have curved edges leading to false detections
        
        PIcutout[:600, 400:] = np.nan

        
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
            overlapcutout[:,0:pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
    
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    elif Mo=='mn1':
        
        # Cutting out the sections with leakage, HII regions, or have curved edges leading to false detections
        
        # PIcutout[800:840, 300:350] = np.nan
        # PIcutout[645:705, 450:500] = np.nan
        # PIcutout[720:810, 770:830] = np.nan
        
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
            overlapcutout[:,0:pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
    
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    
    # Checking if the Mosaic is MO2. One of the observations 
    elif Mo=="mo2":
        
        # The odd source at the top cut out. 
        PIcutout[-350:, 300:850] = np.nan 
        # PIcutout[120:170, 850:900] = np.nan
        # PIcutout[200:325, 150:300] = np.nan
        
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[:pf.mosaic_overlap_width,:] = True
            overlapcutout[:,0:pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
    
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    elif Mo=="mej2":
        
        # The odd source at the top cut out. 
        # PIcutout[508:518, 612:623] = np.nan
        
        
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[:pf.mosaic_overlap_width,:] = True
            overlapcutout[:,0:pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
    
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
            
    # Checking if the mosaic is one on the edge of the mosaic. 
    elif Mo=='meq2' or Mo=='meq1' or Mo=='mel1' or Mo=='mel2':
        
        
        if overlap:
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            if Mo[-1]=="1":
                overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
            else:
                overlapcutout[:pf.mosaic_overlap_width,:] = True 
            if Mo[2]=="q":
                overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            else:
                overlapcutout[:,0:pf.mosaic_overlap_width] = True
            ### Overlapping regions in the mosaics

            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan

     # Checking if the mosaic is a side of the extension  into the halo of the CGPS 
    elif Mo=="me3" or Mo=="me4" or Mo=="me5" or Mo=="mh3" or Mo=='mh4' or Mo=='mh5':

         
         
         if overlap:
             # Creating an array to show the overlap region in the mosaic. The true values 
             #   will be set to NaNs later in the code to show the mask overlap region
             overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
             
             if Mo[-1]=="5":
                 print(Mo[2])
                 overlapcutout[:pf.mosaic_overlap_width,:] = True
             else:
                 overlapcutout[:pf.mosaic_overlap_width,:] = True 
                 overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
             
             if Mo[1]=="e":
                 overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
             else:
                 overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
             ### Overlapping regions in the mosaics
    
             # Cutting out the overlap regions from the polarized intensity image that 
             #   will be used to detect the point sources within it. 
             PIcutout[overlapcutout] = np.nan
    elif Mo=="md2" or Mo=='mh2':
        
    
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions that occur in both mosaics
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            overlapcutout[:,:1*pf.mosaic_overlap_width] = True
            overlapcutout[:1*pf.mosaic_overlap_width,:] = True
            
            # Cutting out the overlap regions that only occur in each mosaic. 
            if Mo=="mh2":
                overlapcutout[-112:,0:694] = True
                
            else:
                overlapcutout[-112:,-554:] = True
                
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    elif Mo=="my2":
        # Cutting out the leakage from the source in the mosaic
        PIcutout[100:300, 200:450] = np.nan
        # PIcutout[992:938, 608:623] = np.nan
        # PIcutout[515:540, 895:915] = np.nan
        
        
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[:1*pf.mosaic_overlap_width,:] = True
            overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    elif Mo=="my1":
        # Cutting out the leakage from the source in the mosaic
        PIcutout[-150:-100, 250:400] = np.nan
        
        
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
            overlapcutout[:,:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan

    elif Mo=="mk1":
        # Cutting out the leakage from the source in the mosaic
        PIcutout[475:800, 150:500] = np.nan
        
        
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
            overlapcutout[:,:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    elif Mo=="mq2":
        # Cutting out the leakage from what I assume is a gas cloud
        PIcutout[450:550, 250:675] = np.nan
        PIcutout[250:450, 375:525]=np.nan
        # PIcutout[250:300, 690:735]=np.nan
        # PIcutout[195:270, 160:300]=np.nan
        
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[:1*pf.mosaic_overlap_width,:] = True
            overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
            
            
    elif Mo=="mer1":
        PIcutout[710:730,627:670] = np.nan
        # PIcutout[550:600,325:400] = np.nan
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
            overlapcutout[:,:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    elif Mo=="mer2":
        
        PIcutout[400:595,360:540] = np.nan
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[:1*pf.mosaic_overlap_width,:] = True
            overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
            
    elif Mo=="mes2":
        
        PIcutout[672:750,282:385] = np.nan
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[:1*pf.mosaic_overlap_width,:] = True
            overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
            
    elif Mo=="mez2":
        
        # PIcutout[920:950,775:805] = np.nan
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[:1*pf.mosaic_overlap_width,:] = True
            overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
            
    elif Mo=="mey1":
        PIcutout[684:753,561:612] = np.nan
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
            overlapcutout[:,:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    elif Mo=="mv1":
        PIcutout[483:522,657:705] = np.nan
        PIcutout[210:270,390:440] = np.nan
        # PIcutout[660:690,615:640] = np.nan
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
            overlapcutout[:,:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
            
    elif Mo=="ml2":
        
        PIcutout[153:195,132:180] = np.nan
        PIcutout[690:,264:732] = np.nan
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[:1*pf.mosaic_overlap_width,:] = True
            overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    elif Mo=="mn2":
        
        PIcutout[900:,132:350] = np.nan
        # PIcutout[150:200,770:830] = np.nan
        
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[:1*pf.mosaic_overlap_width,:] = True
            overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    elif Mo=="mo1":
        PIcutout[759:825,663:732] = np.nan
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
            overlapcutout[:,:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    elif Mo=="mr2":
        
        PIcutout[117:186,285:350] = np.nan
        
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[:1*pf.mosaic_overlap_width,:] = True
            overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
            
    elif Mo=="met2":
        
        # PIcutout[400:475,875:950] = np.nan
        
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[:1*pf.mosaic_overlap_width,:] = True
            overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    elif Mo=="mew2":
        
        # PIcutout[400:475,825:950] = np.nan
        
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[:1*pf.mosaic_overlap_width,:] = True
            overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
            overlapcutout[400:,-1*pf.mosaic_overlap_width:] = True
            overlapcutout[350:400,-1*pf.mosaic_overlap_width +20:] = True
            overlapcutout[:350,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    elif Mo=="ma2":
        
        # PIcutout[500:550,100:150] = np.nan
        
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[:1*pf.mosaic_overlap_width,:] = True
            overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    elif Mo=="mc2":
        
        # PIcutout[150:250,600:700] = np.nan
        # 
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[:1*pf.mosaic_overlap_width,:] = True
            overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    elif Mo=="mf2":
        
        # PIcutout[830:880,100:170] = np.nan
        # PIcutout[728:760,694:735] = np.nan
        # 
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[:1*pf.mosaic_overlap_width,:] = True
            overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
            # overlapcutout[555:590,0:1*pf.mosaic_overlap_width-15] = True
            # overlapcutout[590:,0:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
            
    
    elif Mo=="mej1":

        PIcutout[700:760,330:410] = np.nan
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
            overlapcutout[:,:1*pf.mosaic_overlap_width] = True
            overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    
    elif Mo=="mx1":

        # PIcutout[700:760,330:410] = np.nan
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
            overlapcutout[:,:1*pf.mosaic_overlap_width] = True
            # overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            # A cut out is made in the following so that KR 144 can be detected. 
            overlapcutout[:440,-1*pf.mosaic_overlap_width:] = True
            overlapcutout[440:495,-1*pf.mosaic_overlap_width +30:] = True
            overlapcutout[495:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    
    elif Mo=="mg0":
        if overlap:
            ### Overlapping regions in the mosaics
            
            # Creating an array to show the overlap region in the mosaic. The true values 
            #   will be set to NaNs later in the code to show the mask overlap region
            overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
            
            # Setting the overlap regions to be false so later they will show up when
            #   the mask is plotted. 
            overlapcutout[-379:,:] = True
            # overlapcutout[:,:1*pf.mosaic_overlap_width] = True
            # overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
            
            # Cutting out the overlap regions from the polarized intensity image that 
            #   will be used to detect the point sources within it. 
            PIcutout[overlapcutout] = np.nan
    
    else:
        
        # If there are NaN values in the array, see if in the botton right corner
        #   (indicating the bottom of the mosaic should be cut out) or the top right
        #   corner (indicating the top should be cut out). 
        if np.isnan(PIim1).any():  
            if overlap:
                if np.isnan(PIim1[0,0]):
                    ### Overlapping regions in the mosaics
                    
                    # Creating an array to show the overlap region in the mosaic. The true values 
                    #   will be set to NaNs later in the code to show the mask overlap region
                    overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
                    # Setting the overlap regions to be false so later they will show up when
                    #   the mask is plotted. 
                    overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
                    overlapcutout[:,:1*pf.mosaic_overlap_width] = True
                    overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
                    
                   
    
                    # Cutting out the overlap regions from the polarized intensity image that 
                    #   will be used to detect the point sources within it. 
                    PIcutout[overlapcutout] = np.nan
                elif np.isnan(PIim1[-1,0]):
                    ### Overlapping regions in the mosaics
                    
                    # Creating an array to show the overlap region in the mosaic. The true values 
                    #   will be set to NaNs later in the code to show the mask overlap region
                    overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
                    # Setting the overlap regions to be false so later they will show up when
                    #   the mask is plotted. 
                    overlapcutout[:1*pf.mosaic_overlap_width,:] = True
                    overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
                    overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
    
                   
                    # Cutting out the overlap regions from the polarized intensity image that 
                    #   will be used to detect the point sources within it. 
                    PIcutout[overlapcutout] = np.nan 
                    
                else:
                    print("There are NaNs somewhere in mosaic "+Mo.upper())
                    # Creating a copy array so that they all have the same name 
                    PIcutout= np.copy(PIim1)
            
        
        
        
        
        
        
        
        # Indicating no NaNs were found in the image. This will only apply to the halo mosaics
        else:
            # Creating a copy array so that they all have the same name 
            PIcutout =np.copy(PIim1)
            print("No cut outs were made to mosaic " + Mo.upper())
            if overlap:
                ### Overlapping regions in the mosaics
                
                # Creating an array to show the overlap region in the mosaic. The true values 
                #   will be set to NaNs later in the code to show the mask overlap region
                overlapcutout = np.zeros(shape=PIim1.shape, dtype=bool)
                # Setting the overlap regions to be false so later they will show up when
                #   the mask is plotted. 
                overlapcutout[:1*pf.mosaic_overlap_width,:] = True
                if Mo[-1]!='5':
                    overlapcutout[-1*pf.mosaic_overlap_width:,:] = True
                overlapcutout[:,0:1*pf.mosaic_overlap_width] = True
                overlapcutout[:,-1*pf.mosaic_overlap_width:] = True
                
        
                # Cutting out the overlap regions from the polarized intensity image that 
                #   will be used to detect the point sources within it. 
                PIcutout[overlapcutout] = np.nan
                
                
                
    # Creating an array of values where indicating where there is a number vs a NaN
    maskrev = np.isfinite(PIcutout)
    # Excluding the orignal NaN values from the mask (so we can clearly see
    #    where there is no data)
    maskrev[OGNaNs] = True
    # Creating an initial array that will show be the masked area of the mosaic
    maskforplot = np.ones(shape=(PIim1.shape))
    # Setting the areas where there was a number in the PI to not be part of the mask. 
    maskforplot[maskrev] = np.nan
    
    # Creating an array for the overlap mask to be added to plots, then 
    #    using the previously calculated overlap region (mosaic dependant)
    #    to create the values in the plot. NaNs will not appear in the mask, 
    #    but the zeros will.
    if overlap:
        maskforoverlap = np.zeros(shape=PIim1.shape)
        maskforoverlap[np.invert(overlapcutout)] = np.nan
    
    # Creating an array of values where indicating where there is a number vs a NaN
    maskrev = np.isfinite(PIcutout)
    # Excluding the orignal NaN values from the mask (so we can clearly see
    #    where there is no data)
    maskrev[OGNaNs] = True
    # Cutting out the overlap region so that they don't overlap (important
    #    for matching colormaps)
    if overlap:
        maskrev[overlapcutout] = True
    # Creating an initial array that will show be the masked area of the mosaic
    maskforplot = np.ones(shape=(PIim1.shape))
    # Setting the areas where there was a number in the PI to not be part of the mask. 
    maskforplot[maskrev] = np.nan
    
    # Only returning the overlap cut out if set to. 
    if overlap:
        return PIcutout, overlapcutout, maskforoverlap, maskforplot
    else: 
        return PIcutout, maskforplot
    
# PIcutout, overlapcutout, maskforoverlap, maskforplot, = cut_out_for_mosaic(Pi, "MEZ1")

# plt.imshow(PIcutout)
# plt.show()
# Pi= PIimg("ma1",False)

# # mosaic_edge_cut_out(PI_image, Mo,  plot=False,)
# PIcut, mask = cut_out_for_mosaic(Pi, "ma1",overlap=False)
# t0 = time.time()
# Identify_Point_Sources(PIcut,vmax=0.006)
# t1 = time.time()
# print(f"PI takes {t1-t0} seconds")
# t2 = time.time()
# Identify_Point_Sources(Si, vmax=0.03,threshold=1.5/2000)
# t3 = time.time()
# print(f"Ti takes {t3-t2} seconds")

def Potential_Twin_Finder(Mo, 
                          Plot_twins= True, 
                          plot_individual_sources = False, 
                          return_gal_coord=0,mosaic_overlap=True, PlotPI=False,
                          plot_AGNs = True):
    """ Finds polarised intensity twins for a given mosaic.
    
        This function takes in a mosaic from the CGPS dataset and finds galactic 
        twins or resolved double lobed radio galaxies, and single sources in the 
        polarized intensity image. It requires functions previously defined in 
        the Functions file. It returns the information about the twins, and also 
        creates a plot if desired. 



Key Parameters:
    
        
        Mo (string): 
            The name of the mosaic you wish to go through the detections for. Example for the ME2 mosaic, Mo='me2'

        Plot_twins (boo): 
            Whether to produce a plot of the mosaic with the twin and, if selected, the solo sources. Default is True to produce the plot, selecting False will skip the plot making. 

        
        plot_individual_sources (boo): 
            Whether to plot the solo sources found in the polarized intensity image. 
            
            Default is set to False (not plot them).
                           
        return_gal_coord (int):  
            Whether to return the twin coordinates in galactic coordinates 
            or pixel coordinates. Default is 0 for pixel coordinates 
            only, set to 1 for galactic coordinates only, and 2 for both. 
                       
            Note, the radius
            returned in galactic coordinates is in arcseconds. If this parameter 
            is set to 2 then it will return the pixel coordinates 
            lists first, then the galactic coordinates will follow
            for a total of 6 lists. 
                           
        mosaic_overlap (Boo): 
            Whether to remove the overlap region in the cut out areas 
            of the mosaic. Automatically set to True, which cuts out the overlap region. 
                       
    


 
Returns:
        twinlist (list):
            A 3D list of the each of the twin sources. 
        
            The row or first dimension inticates which set of twins
            you are looking at, the second dimension contains the two twin sources, and the third dimension contains 
            information about the twin source. First it gives the y coordinate of the source, then the x coordinate 
            (coordinate type depends on parameters), then the radius/HWHM (in pixel units or arcseconds). Results in a 
            list that looks like: twinlist = [[[y1,x1,r1], [y2,x2,r2]], ...]. Further description is shown below: 
            twinlist[0] = [twin1,twin2], twinlist[0,0] = twin1 = [y1,x1,r1], twinlist[0,1] = twin2 = [y2,x2,r2], 
            twinlist[0,0,0] = y1. 
        
        distlist (list): 
            1D list containing the distance between the twin sets. 

        twincentres (list): 
            2D list containing the coordinates of the centre of the twins set. This is intended to be used to 
            take a snapshot of the area containing the twins to feed into the next part of the algorithm. 
            twincentres[0]= [y,x] gives the coordinates of the centre of the pair. 

    
Other Parameters:

        PlotPI (boo): 
            Whether to plot the Polarized intensity without the detected sources. 
                      
            Default set to False. 

        plot_AGNs (bool): whether to plot the AGNs from Simbad. 
    
    """
    import parameters_file as pf
    from parameters_file import max_dist_btw_sources as max_dist
    

    t0 = time.time()
    
    
    # beam_radius = 0.5#pixel-ish # the smallest beam size is 1.05 so half that is 0.5
    max_radius = float(pf.max_radius)
    


    # The following library is a python file I made with many functions I thought might use in
    #   different codes. I will probably end putting the detection of twins into a function
    #   or class sometime soon. All the functions have document string with the input 
    #   parameters defined, and what it returns. 
    # import Functions as fc
    
    # Importing the astro functions needed from the astro.py library. 
    from astropy.io import fits
    from astropy.wcs import WCS
    
    # importing function to add circles that indicate either a twin or solo source 
    #   to the legend, and the masked region.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    # Creating the color map for the plot of the polarized and total intensity. 
    mycmap = plt.colormaps.get_cmap("gist_heat")
    mycmap.set_bad(color=pf.NaNcolor)
    
    # Generating the Polarized intensity image. If you are trying to run this you will  
    #   need to input the file directory to the raw CGPS inputs. 
    PIim1 = PIimg(Mo,  
                      plot=PlotPI, return_StoN=False)
   
    
    
    

        
    # Getting the polarised intensity image with the regions cut out that were 
    #   desired to be cut out
    if mosaic_overlap:    
        PIcutout, overlapcutout, maskforoverlap, maskforplot = cut_out_for_mosaic(
            PIim1=PIim1, Mo =Mo, overlap=mosaic_overlap)
    else:
        PIcutout, maskforplot = cut_out_for_mosaic(
            PIim1=PIim1, Mo =Mo,  overlap=mosaic_overlap)
    # t1 = time.time()
    # print("line 1453 first part: {t1-t0}")

    init_sources = Identify_Point_Sources(PIcutout, 
                                        plot=pf.separate_individual_source_plot, threshold=pf.threshold)
    # t2 = time.time()
    # print(f"line 1458 LoG time: {t2-t1}")
    
                
                
                
                
    
    ### Checking if there is a corresponding source in total intensity. 
    ###     This should eliminate a lot of extended source detections. 
    # Creating an empty list to store the coordinates in
    sources_list=[]
    TI_image = T_Inten(Mo, plot=0) # Loading the total intensity image. 
    for s in init_sources: # looping through all the detected sources to get coordinates
        Y,X,r = s
        y,x = int(Y), int(X)
        PI_peak = PIim1[y,x] # finding the peak value in PI
        TI_peak = TI_image[y,x] # find the the peak value in TI
        if TI_peak > pf.ratio_TI_to_PI*PI_peak: # only adding the source if it's sufficiently large in TI
            sources_list += [[y,x,r]]
    # converting the list to an array. 
    sources=np.array(sources_list) 


   
    ### Getting the positional information from the header in galactic coordinates
    
    # Getting the header info from the fits file
    hdu_list = fits.open(pf.img_dir  +Mo+"_1420_MHz_I_image.fits")
    header = hdu_list[0].header
    w = WCS(header)

    
    ticksx = np.linspace(0, len(PIim1[0]), len(PIim1[0]))
    ticksy = np.linspace(0, len(PIim1[:,0]), len(PIim1[0]))
    
    wx, wy, f, meh = w.all_pix2world(ticksx, ticksy,0,0,1)
    
    # Determining the width of a pixel in order to determine the distance between
    #   twin pairs. 
    pixel_width = wx[0]-wx[1]
    # print("line 1516: pixel width", pixel_width)
        
    # Creating a list to store the distance between the twins (I might make the two list one later)
    distlist = []
    # Creating a list to store the coordinates of the twins, and the radii of the 
    #   individual point sources 
    twinlist = []
    # Creating a list of sources that contain the individual sources, this is to avoid double counting
    #   twin sources when false detections occur. 
    twinsources = []
    # Creating a list to store the centre of the twins in so we can make the cutout 
    #   in the future. 
    twincentres=[]
    # Creating a list for the solo sources
    solosources =[]
    # Creating the same lists but these will contain galactic coordinates instead 
    #   of pixel coordinates
    twinlist_galcoord,twinsources_galcoord,twincentres_galcoord,solosources_galcoord, distlist_galcoord = [],[],[],[],[]
    
    # Initialing an array to store the sources in 
    sources_list=[]
    
    # t3 =time.time()
    # print(f"L1521 third part: {t3-t2}")
    
    for n, s in enumerate(sources):
        # Getting the pixel coordinates and HWHM of the source
        Y,X,r = s 
        
        y,x = int(Y), int(X)  # converting the pixel coordinates to intergers
        PI_peak = PIim1[y,x] # finding the peak value in PI
        TI_peak = TI_image[y,x] # find the the peak value in TI
        
        other_sources = np.array(sources.copy())
        
        #Getting the coordinates and HWHM of all other detected sources
        other_sources_y, other_sources_x, other_sources_r = other_sources[:,0].astype(int), other_sources[:,1].astype(int), other_sources[:,2].astype(float)
        dx = other_sources_x - X # Getting the difference in x coordinates between the all the sources and the possible twin source
        dy = other_sources_y - Y # Getting the difference in y coordinates between the all the sources and the possible twin source
        dist_btw = np.sqrt(dx**2+dy**2) # calculating the distance between all the sources
        close_enough = dist_btw <= pf.max_dist_btw_sources # Finding all the sources that are less than the maximum distance apart
        too_close = dist_btw >= pf.min_dist_btw_sources # Finding the sources that are larger than the minimum distance apart
        within_dist_range = close_enough*too_close # finding the sources that are within the range. 
        
        # Setting the values for twin 1
        twin1 = [y,x,r, PI_peak]
        
        # Getting the values in galactic coordinates and in arcsecounds
        twin1_gal_coord = [wy[y], wx[x], r*17.95688, PIim1[y,x]]
        
        ispair = False
        # Getting the indices of the sources within the distance range
        indices_of_possible_pairs = np.nonzero(within_dist_range)[0] 
     
        # if there were any sources in the distance range continue 
        if np.sum(within_dist_range) !=0:
            
        
                
            # if the sources are already in the twin list or aren't the right size
           
            while ispair==False and len(np.nonzero(within_dist_range)[0])!= 0:
                indices_of_possible_pairs = np.nonzero(within_dist_range)[0] 
                # Getting the distance between the sources in the distance range of twin1
                possible_pairs_dist = dist_btw[within_dist_range]
                # Finding the closest sources within the range
                closest_dist = np.min(possible_pairs_dist)
                # getting the index of the closest source. 
                twin2_index = indices_of_possible_pairs[np.argmin(possible_pairs_dist)] 
                
                # Getting the pixel coordinates and HWHM of possible twin2
                y2,x2,r2 = int(other_sources_y[twin2_index]), int(other_sources_x[twin2_index]), float(other_sources_r[twin2_index])
                
                PI_peak
                
                
                # Checking if the peaks are within the right threshold. 
                peaks= [PIim1[y,x], PIim1[y2,x2]]
                peaks.sort(reverse=True) # Putting the list in descending order
                
                ratio_between_PI_peaks = peaks[0]/peaks[1]
                
                within_PI_ratio_threshold = ratio_between_PI_peaks <= pf.ratio_threshold_PI
                
                
                
                # Getting the values and PI peak of twin2
                twin2 = [y2,x2,r2, PIim1[y2,x2]]
                
                # Getting the galactic coordinates, HWHM in acrseconds, and PI of possible twin 2 
                twin2_gal_coord = [wy[y2], wx[x2], r2*17.95688, PIim1[y2,x2]]
                
                # Getting the distance between the possible twins
                dist_btw_twin_pair = dist_btw[twin2_index]
                
                # Only classifying the pair as a twins if: 
                    # both are not in twinsources already
                    # They are the right size
                
                
                if ( twin1 not in twinsources and twin2 not in twinsources 
                    and twin1[2]<max_radius and twin2[2]<max_radius#):
                    and np.round(twin1[2],6)>pf.min_radius and 
                    np.round(twin2[2], 6)>pf.min_radius and
                    within_PI_ratio_threshold == True):
                    
                    # Adding the distance between the pairs to the distance lists
                    distlist.append(dist_btw_twin_pair)
                    distlist_galcoord.append(dist_btw_twin_pair*pixel_width)
                    
                    # Adding the twin pair to the twinliest
                    twinlist.append([twin1, twin2])
                    twinlist_galcoord.append([twin1_gal_coord, twin2_gal_coord])
                    
                    
                    # Adding the each twin to the list of twin sources (this list is to prevent double counting)
                    twinsources.append(twin1)
                    twinsources.append(twin2)
                    
                    # Galactic coordinate version of previous code
                    twinsources_galcoord.append(twin1_gal_coord)
                    twinsources_galcoord.append(twin2_gal_coord)
                    
                    # Finding the center f the twin pairs
                    dY,dX = np.abs((y - y2)), np.abs((x-x2))
                    # Finding the centre of the twins, in the case below the mid x point.
                    if twin1[1] > twin2[1]:
                        centx = twin2[1] + (dX/2)
                    else:
                        centx = twin1[1] + (dX/2)
                    
                    # Finding the y coordinate for the centre of the twins
                    if twin1[0] > twin2[0]:
                        centy = twin2[0] +(dY/2)
                    else:
                        centy = twin1[0] +(dY/2)
                    
                    # Adding the centre of the detected twin in the list
                    twincentres.append([centy, centx]) 
                    
                    
                    
                    dy_gal_coord = twin1_gal_coord[0] - twin2_gal_coord[0]
                    dx_gal_coord = twin1_gal_coord[1]-twin2_gal_coord[1]
                    # Doing the same thing but with galactic coordinates (shortened to galcoord)
                    if twin1_gal_coord[1] > twin2_gal_coord[1]:
                        centx_gal_coord = twin2_gal_coord[1] + (dx_gal_coord/2)
                    else:
                        centx_gal_coord = twin1_gal_coord[1] + (dx_gal_coord/2)
                    
                    # Finding the y coordinate for the centre of the twins
                    if twin1_gal_coord[0] > twin2_gal_coord[0]:
                        centy_gal_coord = twin2_gal_coord[0] +(dy_gal_coord/2)
                    else:
                        centy_gal_coord = twin1_gal_coord[0] +(dy_gal_coord/2)
                    
                    # Adding the centre of the detected twin in the list
                    twincentres_galcoord.append([centy_gal_coord, centx_gal_coord])
                    
                    ispair=True
                
                else:
                    # Getting the index of twin2
                    twin2_index = indices_of_possible_pairs[np.argmin(possible_pairs_dist)]  
                    
                    
                    # Setting twin2 to not be a candidate in the within distance 
                    #   range so we will find the next closest source if any are left. 
                    within_dist_range[twin2_index]=0
                    
                    
                    
                    
                    
            if ispair==False:
                # Adding the source to the solo source list if it's not in the list of twins            
                    if twin1 not in twinsources:
                        solosources.append(twin1)
                        # Adding the info to the galactic coordinates info. 
                        solosources_galcoord.append(twin1_gal_coord)
            # if the sources are already in the twin list or aren't the right size

        
        else: 
            # Adding the source to the solo source list if it's not in the list of twins            
            if twin1 not in twinsources:
                solosources.append(twin1)
                # Adding the info to the galactic coordinates info. 
                solosources_galcoord.append(twin1_gal_coord)
    
    
    
        
            
            
  
    
    
                

            
            
            
  
    # t4 = time.time()
    # print(f"L 1753 loop part: {t4-t3}")
        
            
    
    # printing a list of all the twins detected, and the distance between the two
    if pf.print_coordinates:         
        for i in range(len(twinlist)):
            print("Distance: ", distlist[i], " and pixel coordinates: ", twinlist[i], '\n' +
                 "and the coordinates in galactic coordinates are: ", twinlist_galcoord[i])
        
    if Plot_twins:
        #### Getting Galactic Coordinate stuff/axis together. 
       hdu_list = fits.open(pf.img_dir  +Mo+"_1420_MHz_I_image.fits")
       header = hdu_list[0].header
       w = WCS(header)
       
       ticksx = np.linspace(0, len(PIim1[0]), len(PIim1[0]))
       ticksy = np.linspace(0, len(PIim1[:,0]), len(PIim1[0]))
       
       wx, wy, f, meh = w.all_pix2world(ticksx, ticksy,0,0,1)
       
       # print("pixel_width:", wx[1]-wx[0])
       # print(wx[-1]-wx[0])
       
       
       # Setting the number of ticks to be displayed on the plot
       # tck = [n for n in range(0,1024,pf.num_of_pixels_btw_ticks)]
       tck = [n for n in range(0,len(PIim1[0]),1)]
       # Getting the labels of the x and y ticks
       tickx_labels = np.round(wx[tck],2)
       ticky_labels = np.round(wy[tck],2)
       
       
       
       # Retrieving the total intensity map. 
       OG_TIimage = T_Inten(Mo, plot=0)
       TIimage=np.copy(OG_TIimage)
       
       if Mo == "mg0":
           PI_nans = np.isnan(PIim1)
           TIimage[PI_nans] = np.nan
       
       
       
       # Setting the cut out color maps. 
       edge_cmap= "autumn"
       overlap_cmap="Greys"
       
       # Creating the new plots. Note I need to add the coordinates from the mosaic still
       fig, ax = plt.subplots(1,2, figsize=(16,8), sharex=True, sharey=True)
       # Creating the figure title
       fig.suptitle("Mosaic "+ Mo.upper()[1:], fontsize=1.5*pf.PTF_titlefontsize)

       # Plotting the polarized intensity map            
       Pimage = ax[0].imshow(PIim1, vmin = pf.PI_VMIN, vmax = pf.PI_VMAX,cmap=mycmap,origin='lower')
       # Creating the mask for Polarized intensity plot
       ax[0].imshow(maskforplot, alpha=pf.mask_alpha,cmap= edge_cmap, vmin=0, vmax=1,  origin='lower')
       if mosaic_overlap:
           ax[0].imshow(maskforoverlap, alpha=pf.mask_alpha,cmap= overlap_cmap, vmin=0, vmax=1,  origin='lower')
       ax[0].set_title("Polarized Intensity", fontsize=pf.PTF_titlefontsize)
       ax[0].set_xticks(tck, tickx_labels)
       ax[0].set_yticks(tck, ticky_labels)
       # ax[0].set_xticklabels(tickx_labels)
       # ax[0].set_yticklabels(ticky_labels)
       # ax[0].set_xlim(np.min(tck), np.max(tck))
       # ax[0].set_ylim(np.min(tck), np.max(tck))
       ax[0].set_xlabel(r"Longitude ($\degree$)", fontsize=pf.PTF_axis_font_size)
       ax[0].set_ylabel(r"Lattitude ($\degree$)", fontsize=pf.PTF_axis_font_size)
       fig.colorbar(Pimage, ax=ax[0], label="Jy/beam", shrink=0.74)
       # Plotting the total intensity map 
       
       Timage = ax[1].imshow(TIimage, vmin = pf.TI_VMIN, vmax = pf.TI_VMAX,cmap=mycmap,origin='lower')
       # Creating the mask for Polarized intensity plot
       ax[1].imshow(maskforplot, alpha=pf.mask_alpha,cmap= edge_cmap, vmin=0, vmax=1,  origin='lower')
       if mosaic_overlap:
           ax[1].imshow(maskforoverlap, alpha=pf.mask_alpha,cmap= overlap_cmap, vmin=0, vmax=1,  origin='lower')
       ax[1].set_title("Stokes I", fontsize=pf.PTF_titlefontsize)
       ax[1].set_xlabel(r"Longitude ($\degree$)", fontsize=pf.PTF_axis_font_size)
       ax[1].set_ylabel(r"Latitude ($\degree$)", fontsize=pf.PTF_axis_font_size)
       
       ax[1].set_xticks(tck, tickx_labels)
       ax[1].set_yticks(tck, ticky_labels)
       # ax[1].set_xticklabels(tickx_labels)
       # ax[1].set_yticklabels(ticky_labels)
       # ax[1].set_xlim(np.min(tck), np.max(tck))
       # ax[1].set_ylim(np.min(tck), np.max(tck))
       # axis = ax[1]
       fig.colorbar(Timage, ax=ax[1], label="Jy/beam", shrink=0.74)
       
       # Creating a list of elements to be added to the legend. 
       legend_elements =[Patch(facecolor=pf.Patchcolor_edge, alpha=pf.mask_alpha,
                               edgecolor='black', label="Masked region (region not \nsearched for sources)")]
       legend_elements.append(Patch(facecolor=pf.Patchcolor_overlap, alpha=pf.mask_alpha,
                                    edgecolor='black', label="Overlap region (region not \nsearched for sources)"))
       # Plotting the solo sources if selected to. 
       if plot_individual_sources:
           # Selecting the color of the solo sources. 
           solocolor = 'cyan'
           for s in solosources:
               twiny, twinx, twinr, twinp = s 
           
               solocircle1 = plt.Circle((twinx,twiny), pf.radius_scale*twinr, color=solocolor, linewidth = 2, fill = 0)
               solocircle2 = plt.Circle((twinx,twiny), pf.radius_scale*twinr, color=solocolor, linewidth = 2, fill = 0)
               
               ax[0].add_patch(solocircle1)
               ax[1].add_patch(solocircle2)
               
               
           # Adding the solo source circle to the legend
           legend_elements.insert(0,Line2D([0], [0], color='cyan', 
                  marker = 'o', markeredgewidth=5, linestyle='none', lw=2, label="Solo source",
                  markerfacecolor='w', markersize=15))
    
    
         # Adding the circle to the plot for each Identified twin
       for n, twin in enumerate(twinlist):
           
           # getting the x,y coordinates and radius for each twin
           twinAy, twinAx, twinAr, twinAp = twin[0]
           twinBy, twinBx, twinBr, twinBp = twin[1]
           
           # Creating the circles for plot (for some reason it didn't let me add the 
           #   same circle to two plots). The radius is multiplied by 5 so it's easier 
           #   to see the circle. 
           Color = 'lime'
           # enlarge = 3
           circleA1 = plt.Circle((twinAx, twinAy), pf.radius_scale*twinAr, color=Color, linewidth=2, fill=0)
           circleB1 = plt.Circle((twinBx, twinBy), pf.radius_scale*twinBr, color=Color, linewidth=2, fill=0)
           circleA2 = plt.Circle((twinAx, twinAy), pf.radius_scale*twinAr, color=Color, linewidth=2, fill=0)
           circleB2 = plt.Circle((twinBx, twinBy), pf.radius_scale*twinBr, color=Color, linewidth=2, fill=0)
           
           # Adding the circles around the twins to the previous plots. 
           ax[0].add_patch(circleA1)
           ax[0].add_patch(circleB1)
           ax[1].add_patch(circleA2)
           ax[1].add_patch(circleB2)
       
       # Adding the twins to the top of the legend
       legend_elements.insert(0, Line2D([0], [0], color='lime', lw=2, marker = 'o', markeredgewidth=5, linestyle='none',
                                 label="Identified source in a pair",
                          markerfacecolor='w', markersize=15))
       
       if plot_AGNs:
           tickx_labels_AGN = np.round(wx[tck],3)
           ticky_labels_AGN = np.round(wy[tck],3)
           

           
           lmin = tickx_labels_AGN[-1]
           lmax = tickx_labels_AGN[0]
           bmin, bmax = ticky_labels_AGN[0], ticky_labels_AGN[-1]
           
           # print(get_AGNs_simbad(lmin, lmax, bmin, bmax))
           
           
           AGNs, dfAGNs = get_AGNs_simbad(lmin, lmax, bmin, bmax)
           AGNs = np.round(AGNs, 3)
           print("AGNs:")
           print(AGNs)
           # sys.exit()
           # Adding the SIMBAD AGNs
           if len(AGNs)!= 0:
               AGNs  = np.round(AGNs, 3)
               print("AGNs: \n", repr(AGNs))
               for AGN in AGNs:
                   l, b= AGN
                   
                   x,y  = np.nanargmin(np.abs(tickx_labels_AGN-l)),np.nanargmin(np.abs(ticky_labels_AGN-b))
                   # cir = 0
                   cir1 = plt.Circle((x,y), radius = 2,  color= "magenta", linewidth = 2, fill=0)
                   cir2 = plt.Circle((x,y), radius = 2,  color= "magenta", linewidth = 2, fill=0)
                   ax[0].add_patch(cir1)
                   ax[1].add_patch(cir2)
           
               legend_elements.insert(5, Line2D([0], [0], color='magenta', lw=2, marker = 'o', markeredgewidth=5, linestyle='none',
                                          label="AGN from SIMBAD",
                                   markerfacecolor='w', markersize=15))
    
       # if plot_individual_sources:
       #     box_anchor= (0.922,0.825)
       # else:
       #     box_anchor=(0.80,0.825)
       # # The legend code was based off the code from the webpage: https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html 
       # fig.legend(handles=legend_elements, loc="lower right", 
       #              bbox_to_anchor=box_anchor,bbox_transform=fig.transFigure, 
       #              ncol=4, fontsize='large')
       
       if len(AGNs)==0:
           if plot_individual_sources:
               box_anchor= (0.5,0.88)
           else:
               box_anchor=(0.5,0.8375)
           # The legend code was based off the code from the webpage: https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html 
           fig.legend(handles=legend_elements, loc="center", 
                        bbox_to_anchor=box_anchor,bbox_transform=fig.transFigure, 
                        ncol=4, fontsize='large')
       else:
           if plot_individual_sources:
               box_anchor= (0.5,0.88)
           else:
               box_anchor=(0.5,0.875)
           # The legend code was based off the code from the webpage: https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html 
           fig.legend(handles=legend_elements, loc="center", 
                        bbox_to_anchor=box_anchor,bbox_transform=fig.transFigure, 
                        ncol=5, fontsize='medium')
       
       
       #The following 5 lines allows the axis to update when the window is zoomed in on. 
       ax[0].xaxis.set_major_locator(plt.MaxNLocator("auto"))
       ax[0].yaxis.set_major_locator(plt.MaxNLocator("auto"))
       ax[1].xaxis.set_major_locator(plt.MaxNLocator("auto"))
       ax[1].yaxis.set_major_locator(plt.MaxNLocator("auto"))
       plt.draw()
       fig.tight_layout()
       
       plt.show()
       
           
    if len(twinlist) ==0:
        print("There were no twins detected in mosaic "+Mo.upper())
        
    # t5 = time.time()
    # print(f"L 1923 plotting time: {t5-t4}")
        
    # Determining which if the lists to return. 
    # Just gal_coord
    if return_gal_coord==1:
        return twinlist_galcoord, distlist_galcoord, twincentres_galcoord
    # just pixel 
    elif return_gal_coord == 0:
        return twinlist, distlist, twincentres,
    # both
    else:   
        return twinlist, distlist, twincentres, twinlist_galcoord, distlist_galcoord, twincentres_galcoord

# Potential_Twin_Finder("mf2", plot_individual_sources=False)
# t0 = time.time()
# Potential_Twin_Finder("mf1", plot_individual_sources=True)
# t1 = time.time()
# print(f"It took {t1-t0} seconds")
# Potential_Twin_Finder("mx1", plot_individual_sources=True)



def Manual_Twin_Finder(Mo,mosaic_overlap=True):
    """ This function plots the PI and TI of the entered mosaic, and prints 
        the pixel coordinates wherever the mouse is clicked. 

    Key Parameters:
    
        
        Mo (string): 
            The name of the mosaic you wish to go through the detections for. Example for the ME2 mosaic, Mo='me2'
                           
        mosaic_overlap (Boo): 
            Whether to remove the overlap region in the cut out areas 
            of the mosaic. Automatically set to True, which cuts out the overlap region. 

    
    """
    import parameters_file as pf
    


    # The following library is a python file I made with many functions I thought might use in
    #   different codes. I will probably end putting the detection of twins into a function
    #   or class sometime soon. All the functions have document string with the input 
    #   parameters defined, and what it returns. 
    # import Functions as fc
    
    # Importing the astro functions needed from the astro.py library. 
    from astropy.io import fits
    from astropy.wcs import WCS
    
    # Adding mouse event to the print the xy coordinates in the terminal whenever the mouse clicks somewhere in the image. 
    def mouse_event(event):
        print('x: {} and y: {} in pixel coordinates'.format(np.round(event.xdata,2), np.round(event.ydata)))
    
    
    # Creating the color map for the plot of the polarized and total intensity. 
    mycmap = plt.colormaps.get_cmap("gist_heat")
    mycmap.set_bad(color=pf.NaNcolor)
    
    # Generating the Polarized intensity image. If you are trying to run this you will  
    #   need to input the file directory to the raw CGPS inputs. 
    PIim1 = PIimg(Mo, plot=False, return_StoN=False)
  

        
    # Getting the polarised intensity image with the regions cut out that were 
    #   desired to be cut out
    if mosaic_overlap:    
        PIcutout, overlapcutout, maskforoverlap, maskforplot = cut_out_for_mosaic(
            PIim1=PIim1, Mo =Mo, overlap=mosaic_overlap)
    else:
        PIcutout, maskforplot = cut_out_for_mosaic(
            PIim1=PIim1, Mo =Mo,  overlap=mosaic_overlap)
    ti = time.time()

    
    ### Checking if there is a corresponding source in total intensity. 
    ###     This should eliminate a lot of extended source detections. 
    # Creating an empty list to store the coordinates in
    sources_list=[]


    TI_image = T_Inten(Mo, plot=0) # Loading the total intensity image.
    
    

   
    ### Getting the positional information from the header in galactic coordinates
    
    # Getting the header info from the fits file
    hdu_list = fits.open(pf.img_dir  +Mo+"_1420_MHz_I_image.fits")
    header = hdu_list[0].header
    w = WCS(header)
    
    ticksx = np.linspace(0, len(PIim1[0]), len(PIim1[0]))
    ticksy = np.linspace(0, len(PIim1[:,0]), len(PIim1[0]))
    
    wx, wy, f, meh = w.all_pix2world(ticksx, ticksy,0,0,1)
    
    # Determining the width of a pixel in order to determine the distance between
    #   twin pairs. 
    
        
     #### Getting Galactic Coordinate stuff/axis together. 
    hdu_list = fits.open(pf.img_dir  +Mo+"_1420_MHz_I_image.fits")
    header = hdu_list[0].header
    w = WCS(header)
    
    ticksx = np.linspace(0, len(PIim1[0]), len(PIim1[0]))
    ticksy = np.linspace(0, len(PIim1[:,0]), len(PIim1[0]))
    
    wx, wy, f, meh = w.all_pix2world(ticksx, ticksy,0,0,1)
    
    # Setting the number of ticks to be displayed on the plot
    # tck = [n for n in range(0,1024,pf.num_of_pixels_btw_ticks)]
    tck = [n for n in range(0,len(PIim1[0]),1)]
    # Getting the labels of the x and y ticks
    tickx_labels = np.round(wx[tck],2)
    ticky_labels = np.round(wy[tck],2)
    
    
    
    # Retrieving the total intensity map. 
    TIimage = T_Inten(Mo, plot=0)
    

    
    # Setting the cut out color maps. 
    edge_cmap= "autumn"
    overlap_cmap="Greys"
    
    # Creating the new plots. Note I need to add the coordinates from the mosaic still
    fig, ax = plt.subplots(1,2, figsize=(16,8))
    
    cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
    # Creating the figure title
    fig.suptitle("Mosaic "+ Mo.upper(), fontsize=1.5*pf.PTF_titlefontsize)

    # Plotting the polarized intensity map            
    Pimage = ax[0].imshow(PIim1, vmin = pf.PI_VMIN, vmax = pf.PI_VMAX,cmap=mycmap,origin='lower')
    # Creating the mask for Polarized intensity plot
    ax[0].imshow(maskforplot, alpha=pf.mask_alpha,cmap= edge_cmap, vmin=0, vmax=1,  origin='lower')
    if mosaic_overlap:
        ax[0].imshow(maskforoverlap, alpha=pf.mask_alpha,cmap= overlap_cmap, vmin=0, vmax=1,  origin='lower')
    ax[0].set_title("Polarized intensity", fontsize=pf.PTF_titlefontsize)
    
    ax[0].set_xticks(tck, tickx_labels)
    ax[0].set_yticks(tck, ticky_labels)
    
    ax[0].set_xlabel("Longitude", fontsize=pf.PTF_axis_font_size)
    ax[0].set_ylabel("Lattitude", fontsize=pf.PTF_axis_font_size)
    fig.colorbar(Pimage, ax=ax[0], label="Jy/beam", shrink=0.74)
    # Plotting the total intensity map 
    
    Timage = ax[1].imshow(TIimage, vmin = pf.TI_VMIN, vmax = pf.TI_VMAX*1,cmap=mycmap,origin='lower')
    # Creating the mask for Polarized intensity plot
    ax[1].imshow(maskforplot, alpha=pf.mask_alpha,cmap= edge_cmap, vmin=0, vmax=1,  origin='lower')
    if mosaic_overlap:
        ax[1].imshow(maskforoverlap, alpha=pf.mask_alpha,cmap= overlap_cmap, vmin=0, vmax=1,  origin='lower')
    ax[1].set_title("Total intensity", fontsize=pf.PTF_titlefontsize)
    ax[1].set_xlabel("Longitude", fontsize=pf.PTF_axis_font_size)
    ax[1].set_ylabel("Latitude", fontsize=pf.PTF_axis_font_size)
    ax[1].set_xticks(tck, tickx_labels)
    ax[1].set_yticks(tck, ticky_labels)
    
    fig.colorbar(Timage, ax=ax[1], label="Jy/beam", shrink=0.74)
    
    plt.tick_params('y', labelleft=True)
    #The following 5 lines allows the axis to update when the window is zoomed in on. 
    ax[0].xaxis.set_major_locator(plt.MaxNLocator("auto"))
    ax[0].yaxis.set_major_locator(plt.MaxNLocator("auto"))
    ax[1].xaxis.set_major_locator(plt.MaxNLocator("auto"))
    ax[1].yaxis.set_major_locator(plt.MaxNLocator("auto"))
    plt.draw()
    
    
    
    fig.tight_layout()
    plt.draw()
    # plt.show()
           
   
    
# mosaics = ["mk2", "mh1", "mo1", "mew2", "mr2",'mex2', 'mu2', 'mh1']
# Manual_Twin_Finder("met1")                
 
def list_mosaics(direction="LtoR", include_halo=True, include_SLE = True):
    "A function to give a list of mosaics. Direction is either LtoR (left to right) or RtoL (right to left)"
    list_mosaics = []
    mosaic_columns = ["meq", "mer", "mes", "met", "meu", "mev", "mew", "mex", "mey",
                      "mez","mst", "mu", "mv", "mw", "mx", "my","ma", "mb", "mc","md",
                      "me", "mf", "mg", "mh", "mij", "mk","ml", "mm", "mn", "mo", "mp", 
                      "mq", "mr", "mej", "mek", "mel"]

    for m in mosaic_columns:
        top,bottom  = m +"2", m +"1"
        list_mosaics.append(bottom)
        list_mosaics.append(top)
    halo =[]
    halo_columns = ["me", "mf", "mg", "mh"]
    for h in halo_columns:
        bottom, middle, top = h+"3", h+"4", h+"5" 
       
        halo += [bottom] +[middle]+[top]
        
    if include_halo:
        list_mosaics+=halo

    if include_SLE:
        list_mosaics+=["mg0"]
    if direction=="LtoR":
        return list_mosaics
    else:
        return list_mosaics.reverse()
# print(list_mosaics())
# print(len(list_mosaics()))


def random_mosaics(n=3, exclude=None):
    """This function returns a random list of mosaics. This is literally just so 
    I can don't have randomly choose them myself. 
    
    Parameters: 
        n: the number of mosaics to return (default is 3)
        exclude: The list of mosaics to include. (default is None)
    Returns: random_mosaics the list of random mosaics
        """
    
    from random import choice
    
    mosaics = list_mosaics()

    if exclude !=None:
        for m in exclude:
            mosaics.remove(m)
    random_mosaics = []
    for i in range(n):
        rdm_mosaic = choice(mosaics)
        random_mosaics += [rdm_mosaic]
        mosaics.remove(rdm_mosaic)

    return random_mosaics

# print(random_mosaics(1))

# mosaics = ["mk2", "mh2", "mo1", "mew2", "mr2"]
def binary_classification_dictionaries():
    
    
    dictionary_words = {None: "No sources were detected in the cutout.",
                        
                  17: "One solo source in Stokes I cutout was correlated with the two in polarised intensity.",
                  9: "One source was detected in the Stokes I cutout but was not correlated with the sources in Polarized intensity.",
                  34: "Two sources detected in, and two sources in the correlation region, but the Stokes I sources are too close together.",
                  546: "Twin detected, two sources were detected in Stokes correlated to sources in polarized intensity. Twin detected",
                  18: "Two sources were detected in Stokes I but only one was correlated to the sources in polarized intensity",
                  10: "Two sources were detected in Stokes I but neither correlated to the sources in polarized intensity.",
                  146: "Two sources were detected but were too far apart to the be a twin source, but one source correlated to the pair in polarized intensity.",
                  162: "Two sources were detected but were too far apart to be twins, but both sources correlated to the sources in polarized intensity",
                  138: "Two sources were detected but neither of them are correlated to the sources in polarised intensity.",
                  12: "Multiple sources were detected but none were within the correlation region.",
                  20: "Multiple sources were detected in Stokes I, but only one of the sources was within the correlation region.",
                  548: "Multiple sources were detected in Stokes I, but only two were in the correlation region. Twin detected.",
                  164: "Multiple sources were detected in Stokes I, but only two were in the correlation region but they were too fair apart to be a twin.",
                  580: "Twin detected. Multiple sources were detected in Stokes I, multiple were in the correlation region and the two closest sources were in the correlation region.",
                  196: "Multiple sources were detected in Stokes I, multiple were in the correlation region but the two closest sources to those in PI were too far apart.",
                  False: "A false detection occured."
                  }
    # list = [number of detected sources, sources within the cor region, twin detected, sources close enough together,]
    dictionary_array = {None:[None, None, False],
                      17: [1, 1, False],
                      9:  [1,0,False],
                      34:[2,2,False],
                      546: [2,2, True],
                      18: [2, 1, False],
                      10: [2,0, False],
                      146: [2, 1, False],
                      162: [2, 2, False],
                      138: [2,0, False],
                      12: [3, 0, False],
                      20: [3, 1, False],
                      548:[3, 2,True],
                      164: [3, 2, False],
                      196: [3, 3, False],
                      580: [3, 3, True],
                      False: [None, None, None]
                      }
    
    return dictionary_words, dictionary_array

def beam_radii(mosaic, width_in_arcminutes=False):
    """Function gives the beam radi (Default in pixel length) for a given declination 
    of a mosaic in the CGPS data.
    
    Uses the fact that one pixel in the CGPS is approximately 17.95688'', and the
    beam size is given by (49'')^2*cosec(δ), where is the dec in degrees. 
    
    Parameters:
        dec: The declination in degrees
        
        width_in_arcminutes (boo): Return the radius in arcminutes or pixel length.
            Default is false for pixel length.
        
    Returns:
        radi: radius of the beam (either pixel or arcminutes based on parameters.)
            
    """
    from astropy.io import fits
    import parameters_file as pf
    ####### pixel width calculation:
        # pw = wx[0]-wx[1] = 0.0049883008487086045°
        # pw_arcsec = pw(3600"/1°)  = 17.957883055350976"
        # CGPS paper says 18"/pixel so this is probably correct
    
    ###### beam width range #######
        # 2.4153724105756993 (beam in mel2 at dec 18.885714) to
        # 1.056896887420664 (beam in mb2 at dec 65.542709)
        
    #path for total intensity, this is the same for all bands so it only needs  
    #total intensity
    hdu_listI = fits.open(pf.img_dir  +mosaic+"_1420_MHz_I_image.fits")
    
    #Getting the mosaic information for the coordinates 
    headerI = hdu_listI[0].header
    
    dec = headerI[64]
    
    print("dec: ", dec)
    cosec = 1/np.cos(np.radians(dec))
    print("cosec: ", cosec)
    
    
    numerator = (58**2)/(18**2)
    # print("reverse: ", np.arccos(numerator/3.3))
    beam_size= (numerator)/(np.sin(np.radians(dec)))
    
    print("beam_size: ", beam_size, " pixels")
    
    radi = np.sqrt(beam_size/np.pi)
    if width_in_arcminutes:
        return radi*17.95688
    
    radi_pixel= radi#/17.95688
    return radi_pixel

# Note this was taken from subroutines in Jo-Anne and Cameron's code, converted by Dylan.  
def string_normalise(string_bad, expected_len, negatives=False, front_load=False):
    """This function normalises the length of a string for ease of printing to a file, meaning if the length of the string
    is less than the expected length, it will add whitespace after the string until the length of the string equals the
    expected length.
    For example:
    '-131.1' has a length of 6, but should have a length of 8. This function then outputs '-131.1  '

    ARGUMENTS:
    - string_bad (string) -- the un-normalised string
    - expected_len (int)  -- the expected length of the string, must be at least equal to the length of string_bad
    - negatives (bool)    -- an optional condition which allows for further normalising of numbers that may have a negative sign
    - front_load (bool)   -- an optional condition which, if true, will add the whitespace to the beginning of the string instead of the end

    RETURNS:
    - string_good (string) -- the normalised string
    """
    if string_bad[0] != '-' and negatives:
        string_bad = ' ' + string_bad
    string_len = len(string_bad)
    # print("string_length: ", string_len)
    len_diff = expected_len - string_len  # expected_len must be at least as large as string_len
    whitespace = ' ' * len_diff  # Interestingly, Python lets you multiply strings by ints and it just repeats the string
    if front_load:
        string_good = whitespace + string_bad
    else:
        string_good = string_bad + whitespace
    return string_good



# Note this was taken from subroutines in Jo-Anne and Cameron's code, converted by Dylan.
def nround(number):
    """As I've recently learned, the built in Python round() and int() functions
    both round .5s down instead of up, so round(2.5) = 2, when it should really be 3.

    This makes no sense, and so I need to implement my own rounding function.

    ARGUMENTS:
    - number (float or int) -- the number to be rounded up or down to an int

    RETURNS:
    - the rounded number in integer form
    """
    if number - math.floor(number) < 0.5:
        return math.floor(number)
    else:
        return math.ceil(number)
    
    
def find_pixel_coordinates(long, lat, mosaic):
    """This function finds the pixel coordinate of a point in galactic coordinates.
    
    Parameters)
        - long: (float) the longitute of the point
        - lat: (float) the latitude of the source
        - mosaic: (str) the mosaic the source is in. 
        
    Outputs
        - x: (int) x pixel coordinate
        - y: (int) y pixel coordinate"""
    from astropy.io import fits
    from astropy.wcs import WCS
    import parameters_file as pf
     
    #total intensity
    hdu_listI = fits.open(pf.img_dir  +mosaic+"_1420_MHz_I_image.fits")
        
    #Getting the mosaic information for the coordinates 
    headerI = hdu_listI[0].header
       
    # print(headerI)
    # print(repr(headerI))
    # getting and removing unnecessary dimensions from the data
    imI = np.squeeze(hdu_listI[0].data)


    # Adding the galactic coordinates to the image, the coordinates will 
    # not change between the files so any header can be used for this 
    w = WCS(headerI)
    #making an array with the number of pixels in the image
    ticksx = np.linspace(0, len(imI[0]), len(imI[0]))
    ticksy = np.linspace(0, len(imI[:,0]), len(imI[0]))
    
    #Using the information from the header and the number of pixels determining 
    # what the coordinates of the image is 
    wx, wy, f, meh = w.all_pix2world(ticksx, ticksy,0,0,1)
    
    # Setting the number of ticks to be displayed on the plot
    tck = [n for n in range(0,1024)]
    # Getting the labels of the x and y ticks
    tickx_labels = np.round(wx[tck],2)
    ticky_labels = np.round(wy[tck],2) 
    
    unrounded_x = wx[tck]
    unrounded_y = wy[tck]
    
    # print(tickx_labels)
    # print(long)

    x = np.where(tickx_labels == round(long,2) )[0].tolist()#+0.01 and tickx_labels.any() > long-0.01)
    y = np.where(ticky_labels == round(lat, 2))[0].tolist()
    
   
    if len(x)>2:
        print("Huston we have a problem")
    
    if unrounded_x[x[0]] - long < unrounded_x[x[1]] - long:
        X = x[0]
    else:
        X = x[1]
    
    if unrounded_y[y[0]] - lat < unrounded_y[y[1]] - lat:
        Y = y[0]
    else:
        Y=y[1]
        
    print(f"The pixel coordinates are: \n\t{X} in the x direction \n\t{Y} in the y direction")
    
    return X,Y

def find_gal_coordinates(x, y, mosaic):
    """This function finds the galatic coordinates of a point in pixel coordinates.
    
    Parameters)
        - x: (int) x pixel coordinate
        - y: (int) y pixel coordinate
        - mosaic: (str) the mosaic the source is in. 
        
    Outputs
        - long: (float) the longitute of the point
        - lat: (float) the latitude of the source
        """
    from astropy.io import fits
    from astropy.wcs import WCS
    import parameters_file as pf
     
    #total intensity
    hdu_listI = fits.open(pf.img_dir  +mosaic+"_1420_MHz_I_image.fits")
        
    #Getting the mosaic information for the coordinates 
    headerI = hdu_listI[0].header
       
    # print(headerI)
    # print(repr(headerI))
    # getting and removing unnecessary dimensions from the data
    imI = np.squeeze(hdu_listI[0].data)


    # Adding the galactic coordinates to the image, the coordinates will 
    # not change between the files so any header can be used for this 
    w = WCS(headerI)
    #making an array with the number of pixels in the image
    ticksx = np.linspace(0, len(imI[0]), len(imI[0]))
    ticksy = np.linspace(0, len(imI[:,0]), len(imI[0]))
    
    #Using the information from the header and the number of pixels determining 
    # what the coordinates of the image is 
    wx, wy, f, meh = w.all_pix2world(ticksx, ticksy,0,0,1)
    
    # Setting the number of ticks to be displayed on the plot
    tck = [n for n in range(0,1024)]
    
    
    unrounded_x = wx[tck]
    unrounded_y = wy[tck]
   
    X = np.round(unrounded_x[x],3)
    Y = np.round(unrounded_y[y],3)
    print(f"The pixel coordinates are: \n\t{X} degrees longitude \n\t{Y} degrees latitude")
    
    return X, Y
    
# find_gal_coordinates(349,670,"mf2")

def l_b_TI_PI_SN_of_point(xcoor, ycoor, mosaic):
    """This function takes in the x and y coordinates of a point in the given mosaic
    and returns the gal. longitude, the gal. latitude, the Total Intensity, the 
    Polarized Intensity, and the signal to noise of that point.
    Please note: if the x and y coordinates given are pixel coordinates, please 
    ensure they are integer values. 
    
    Parameters:
        
        - xcoord: the x coordinate of the source. If the entered source is an 
        integer value then it assumed to be a pixel value, if float it is assumed
        to be galactic coordinates. 
        
        - ycoord: the y coordinate of the source. If the entered source is an 
        integer value then it assumed to be a pixel value, if float it is assumed
        to be galactic coordinates. 
        
        - mosaic: the mosaic the point is located in. 
    Returns:
        l (float): the longitute of the point 
        
        b (float): the latitude of a point
        
        xpix (int): The x pixel value of the point
        
        ypix (int): The y pixel value of the point 
        
        PI (float): the polarized intensity of the point (mJy/beam)
        
        TI (float): the total intensity of the source (mJy/beam)
        
        SN (float): the signal to noise of the source in PI.
        """
        
    if type(xcoor) == int and type(ycoor)==int:
        xpix = xcoor
        ypix = ycoor
        
        l,b = find_gal_coordinates(xpix, ypix, mosaic)
        
    else:
        l,b = xcoor, ycoor
        
        xpix, ypix = find_pixel_coordinates(l, b, mosaic)
    PI_mosaic, SN_mosaic = PIimg(mosaic, plot=False, return_StoN=True)
    TI_mosaic = T_Inten(mosaic, plot=False) 
    
    PI, TI, SN = PI_mosaic[ypix, xpix]*1000, TI_mosaic[ypix, xpix]*1000, SN_mosaic[ypix, xpix] #The *1000 to convert from Jy to mJy
    
    return l, b, xpix, ypix, PI, TI, SN


    
def write_dat_file(mosaic, point_dataset):
    """This function writes the data file for twins detected in the algorithm. 
    Specifically, it will write a dat file that Jo-Anne/Camerons RM code can read. 
    If the folder for the mosaic does not already exist, the program will create a new one. 
    
    Inputs:
        mosaic (str): name of the mosaic. 
        mosaic_dataset (2D list): the list of all the values from the dataset 
            of the mosaic chosen. """
            

    # Loading where to write the dat file to 
    from directories import RM_out_dir
    # Importing the function that allows you to create a folder in a directory 
    from os import makedirs
    
    #Converting all the mosaic dataset to an array 
    point_array = np.array(point_dataset)
    
    # Creating the folder name. The RM code requires the name not include the 
    #   "m" at the start of the folder name. So "mf3" needs to be named "f3".
    #   The RM code also requires the dat file for each mosaic to be it's own 
    #   separate folder.
    if mosaic[0] != 'm' and mosaic[0] != "M":
        foldername = mosaic.lower()
    else:
        foldername = mosaic[1:].lower()
        
    # Setting the output directory of the dat file. 
    out_dir = f'{RM_out_dir}{foldername}'
    
    # Making a new folder 
    makedirs(out_dir, exist_ok=True)
    
# =============================================================================
#     Getting values from dataset
# =============================================================================
    shape=  point_array.shape
    
    #old version
    # if len(shape) == 1:
    #     single_point = True
    #     l, b, xpix, ypix, PI, TI, SN = point_dataset[0],point_dataset[1],point_dataset[2],\
    #         point_dataset[3],point_dataset[4],point_dataset[5],point_dataset[6],
    #     # print("line 4143")
    #     # sys.exit()
    # else:
    #     single_point = False
    #     l, b, xpix, ypix, PI, TI, SN = point_array[:,0], point_array[:,1],\
    #     point_array[:,2],point_array[:,3],point_array[:,4],point_array[:,5],\
    #         point_array[:,6]
    
    if len(shape) == 1:
        single_point = True
        l, b, xpix, ypix, PI, TI, SN = point_dataset[0],point_dataset[1],point_dataset[2],\
            point_dataset[3],point_dataset[4],point_dataset[5],point_dataset[6],
        # print("line 4143")
        # sys.exit()
    else:
        single_point = False
        l, b, xpix, ypix, PI, TI, SN = point_array[:,0], point_array[:,1],\
        point_array[:,2],point_array[:,3],point_array[:,4],point_array[:,5],\
            point_array[:,6]
    
    
    # print("l: ", l)
# =============================================================================
#     # Writing the file
# =============================================================================
    with open(f'{out_dir}/{mosaic.upper()}_points.dat', "w") as write_points:
        write_points.write(f'Polarized twin source candidate list for field {mosaic.upper()}')
        write_points.write('\nGenerated using Ciara Chisholms twin source detection algorithm')
        write_points.write('\n ')
        write_points.write('\n    l        b    xpix  ypix       PI       SI      S/N')
        write_points.write('\n--  degrees   --                mJy/beam mJy/beam')
        write_points.write('\n ')
        
        if single_point:
            write_points.write(f'\n{string_normalise(str(round(l, 3)), 10)}'
                              f'{string_normalise(str(round(b, 3)), 9, negatives=True)}'
                              f'{string_normalise(str(nround(xpix)), 6)}'
                              f'{string_normalise(str(nround(ypix)), 9)}'
                              f'{string_normalise(str(round(PI, 2)), 9)}'
                              f'{string_normalise(str(round(TI, 2)), 8)}'
                              f'{string_normalise(str(round(SN, 2)), 5)}')
        else: 
            for p, L in enumerate(l):
                write_points.write(f'\n{string_normalise(str(round(l[p], 3)), 10)}'
                                  f'{string_normalise(str(round(b[p], 3)), 9, negatives=True)}'
                                  f'{string_normalise(str(nround(xpix[p])), 6)}'
                                  f'{string_normalise(str(nround(ypix)[p]), 9)}'
                                  f'{string_normalise(str(round(PI[p], 2)), 9)}'
                                  f'{string_normalise(str(round(TI[p], 2)), 8)}'
                                  f'{string_normalise(str(round(SN[p], 2)), 5)}')
       
    print(f'\nTwins pointlist generated for mosaic {mosaic.upper()}!')

def pixel_to_galactic_coordinates_axis(mosaic):
    
    """This function returns an array of the galactic coordinates of the fits image,
    specifically it takes in returns the tick values of mosaic in galactic coordinates.
    
Parameters:
    
    mosaic (str): 
        the mosaic you are looking to get coordinates for.
        
    img_dif (str): 
        where the mosaic fits file is located. 
        
Returns:
    
    wx (1D array): 
        the x coordinates for every pixel in the image.
    wy (1D array): 
        the y coordinates for every pixel in the image."""
        
    # Importing the necessary functions
    from astropy.io import fits
    from astropy.wcs import WCS
    from parameters_file import img_dir
    
    
    
    #Opening the fits file. 
    hdu_listI = fits.open(img_dir  +mosaic+"_1420_MHz_I_image.fits")
        
    #Getting the mosaic information for the coordinates from the header
    headerI = hdu_listI[0].header
    # print(repr(headerI))
    #   getting and removing unnecessary dimensions from the data
    imI = np.squeeze(hdu_listI[0].data)

    # Adding the galactic coordinates to the image, the coordinates will 
    # not change between the files so any header can be used for this 
    w = WCS(headerI)
    
    #making an array with the number of pixels in the image
    ticksx = np.linspace(0, len(imI[0]), len(imI[0]))
    ticksy = np.linspace(0, len(imI[:,0]), len(imI[:,0]))
    
    #Using the information from the header and the number of pixels determining 
    # what the coordinates of the image is 
    wx, wy, f, meh = w.all_pix2world(ticksx, ticksy,0,0,1)

    return wx, wy
     

def GAL_to_RADEC(L,B, in_deg = False):
    """Converts gal coordinates to RA and DEC. 
    
    paramters: 
        L (float): the longitude of the source
        B (Float) : the latitude
        in_deg (bool): whether to return in degrees, or (the default) RA in hours minutes seconds 
            and DEC in degress minutes seconds.
    Returns:
        RA (String)
        DEC (String) """
    
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    # gal_cor = SkyCoord(l = L*u.deg, b=B*u.deg, frame='galactic')
    
    # equatorial_cor = gal_cor.transform_to("icrs")
    new_coord = SkyCoord(l=L*u.degree, b=B*u.degree, frame="galactic")
    ICRS = new_coord.icrs
    ra, dec = ICRS.ra, ICRS.dec
    ra_str, dec_str = ra.to_string(unit=u.hourangle, sep= " ", precision=2), dec.to_string(sep=" ", precision=2)
    
    if in_deg: 
        return ra.deg, dec.deg
    else:
        return ra_str, dec_str          


def check_true_detection(detected_twins_df):
    """This function takes in the dataframe produced by the twin detection algorithm 
    per mosaic and returns a dataframe that says if the detections were real or false. 
    
    Note: pandas must be imported as pd to run. 
    
    Parameters:
        mosaic_df (dataframe): The dataframe with the twin information from the twin detector algorithm (all mosaics),
            only twins though. 
        
        true_twin_detected_lst (array): This is an array the length of the true twins list, 
            and is true if the twin corresponding to that row has been detected, and is False
            if it has not been. Default value is None, and the code creates a array that length
            full of False values. 
            
        Returns:
            new_df (dataframe): the mosaic_df with the true values included. Specifically
                column 2 is the 'True twin detection' column, and column 4 is the 
                'True Class' column. 
            missed_twin_df (df): the dataframe containing all the missed twins information. 
            """
            
            
    #importing the path to the true twins. 
    from directories import true_twin_csv_dir 
    
    # importing the true twins as a dataframe
    true_df = pd.read_csv(true_twin_csv_dir)#, index_col="n")
    
    #Pandas imports empty values as NaNs, the next little bit of code is converting them back empty or None values
    true_df = true_df.fillna("").copy()
    detected_twins_df = detected_twins_df.fillna("").copy()
   

    # Getting the galactic coordinates of the center of the true twins
    true_center_l, true_center_b = true_df["Center l (deg)"].to_numpy(), true_df["Center b (deg)"].to_numpy()
    
    
    # Creating an array to keep track of which true twins have been detected. Initially setting none of them to have been detected. 
    true_twin_detected_from_csv = np.zeros(shape = (len(true_center_b)), dtype=bool)#np.full(shape = (len(true_df)), fill_value = False)
    
    # Getting the galactic coordinates of the center of the detected twins
    center_l_detected, center_b_detected = detected_twins_df["Center l (deg)"].to_numpy(), detected_twins_df["Center b (deg)"].to_numpy()
    
    # Initializing lists to store the "True" column value 
    true_twin_classes_lst,true_twin_detected_lst = [], []
    
    detected_twins_df_index = detected_twins_df.index.tolist()
    
    # Setting the maximum distance the true twins and the detected twins can 
        # be away from each other in case there is slight difference in locations but it is still the same soures being detected
    max_dist_away = np.sqrt(2)*(0.01)# 0.0051 degrees squared is from having a one pixel correlation region and 1pix = 0.005 degrees
    
    
    # Looping through every detected twin
    for n, l in enumerate(center_l_detected):
        # finding the offset in coordinates between the detected twin and the true twins 
        dl_array = true_center_l - center_l_detected[n]
        db_array = true_center_b - center_b_detected[n]
  
        # Finding the disance between twins
        dist_from_true_twins = np.sqrt(dl_array**2+db_array**2)
        
        # Finding the smallest distance to any twins, min index corresponds to which true twin is closest
        min_index = np.argmin(dist_from_true_twins)
  
        
        # Finding the minimum distance between the twins 
        min_dist_from_true_twins  = dist_from_true_twins[min_index]
        
        
        # below until the end of the if statement Checks if they are the right sources because that one false detection in ME4
        true_twins_gal_long = np.sort([true_df.loc[min_index, "Galactic Longitude of twin 1 in PI (degrees)"], true_df.loc[min_index, "Galactic Longitude of twin 2 in PI (degrees)"]])

        det_twins_gal_long = np.sort([detected_twins_df.loc[detected_twins_df_index[n], "Galactic Longitude of twin 1 in PI (degrees)"], detected_twins_df.loc[detected_twins_df_index[n], "Galactic Longitude of twin 2 in PI (degrees)"]])
       
        if min_dist_from_true_twins <= max_dist_away:
            if np.abs(true_twins_gal_long[0]- det_twins_gal_long[0]) >= 0.006 or \
                    np.abs(true_twins_gal_long[1]- det_twins_gal_long[1]) >= 0.006:#failing if the detected twins aren't close enough together to the true ones. 
                min_dist_from_true_twins=100

        
         
        
        if min_dist_from_true_twins <= max_dist_away:
            # Finding the true class of the twin
            true_twin_class = true_df.at[min_index, "True Class"]
            # adding the true class to the list 
            true_twin_classes_lst.append(true_twin_class)
            # adding that a true twin was detected to the true twin detected list
            true_twin_detected_lst.append(True)
            # 
            true_twin_detected_from_csv[min_index] = True
        else:
            true_twin_classes_lst.append(False)
            true_twin_detected_lst.append(False)
             
    # copying the mosaic database 
    new_df = detected_twins_df.copy()
    
    new_df.insert(2, "True twin detection", true_twin_detected_lst)

    new_df.insert(4, "True Class", true_twin_classes_lst)
    
    
    missed_twins_array = ~np.array(true_twin_detected_from_csv, dtype=bool)
    print("L3322) missed twins: ", np.sum(missed_twins_array))
    
    
    if np.sum(missed_twins_array)>0:
        
        
        missed_twins_df = true_df[missed_twins_array].copy()
        
        print(np.where(missed_twins_array)[0])
        print(true_df["Mosaic"].to_numpy()[missed_twins_array])
        missed_twins_df.loc[:,"Twin detected"] = False
        
        missed_twins_df.loc[:,"Class"] = 0
        
        all_twins_df = pd.concat((new_df, missed_twins_df), ignore_index=True)
        return all_twins_df, missed_twins_df
    else:
        return new_df, pd.DataFrame([])
    
    
    

# point_info = np.array(TI_PI_SN_of_point(109.569, 3.788, "mf2"))   
# print(point_info[0]) 

# write_dat_file("mf2", point_info)


        
# find_pixel_coordinates(109.570,3.785,"mf2")
    
# print("me2", beam_radii("me2"), "\n")
# print("mel2", beam_radii("mel2"))
# mo = "ma2"
# mosaics = []
# for i in range(1,2):
#     mosaics += [mo + str(i)]
# for mo in mosaics:
#     print(mo)
# PIimg(mo, gal_coord=False)
# tl, dl, tc = Potential_Twin_Finder(mo, Plot_twins=True, plot_individual_sources=True, PlotPI=False, remove=0)

# plt.pause(1)

# mo="mv2"
# tl, dl, tc = Potential_Twin_Finder(mo, Plot_twins=True, plot_individual_sources=True,
#                                 PlotPI=False)
    # plt.pause(1)


# mosaic_columns = ["meq", "mer", "mes", "met", "meu", "mev", "mew", "mex", "mey",
#                   "mez","mst", "mu", "mv", "mw", "mx", "my","ma", "mb", "mc",
#                   "me", "mf", "mg", "mh", "mij", "mk", "mm", "mn", "mo", "mp", 
#                   "mq", "mr", "mej", "mek", "mel"]
# print(len(mosaic_columns))

# PIim1 = PIimg("mo2", plot=False)
# sources = Identify_Point_Sources(PIim1,plot=True, )

# TI = T_Inten("mo2", plot=1)


# Mf2 had the different radii detection and it has things classed as twins that 
#   I think are too far apart to be actual twins. 
# MF1 appears to only have false detections 
# tl, dl, tc = Potential_Twin_Finder('mey2', plot_individual_sources=True)
# piimage = T_Inten('mx1')

# mosaics = list_mosaics()

# n = 0
# for M in mosaics:
#     print("Writing ", M, " iteration ", n)
#     PIimg(mosaic=M, plot=False,  create_fits=True,new_img_dir ="/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/CGPS2012/PI_fits/")
#     n+=1
    
    
    