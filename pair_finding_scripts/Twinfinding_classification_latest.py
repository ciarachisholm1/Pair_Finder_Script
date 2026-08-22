#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:59:26 2023

@author: ciarachisholm
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 11:20:22 2023

@author: ciarachisholm
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time 

import Functions as fc
from RM_StoN import RM_code_StoN
from astropy.io import fits
from astropy.wcs import WCS


##### Note to self: the pixel width in the CGPS is about 0.004988 degrees or 17.9568" (arcseconds)




################################################################
############################# Code #############################
################################################################

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





    
def Potential_Twin_Finder(Mo, 
                          Plot_twins= True, 
                          plot_individual_sources = False, 
                          return_gal_coord=0,mosaic_overlap=True, PlotPI=False,
                          plot_AGNs=True):
    """ Finds polarised intensity twins, or pairs, for a given image (or mosaic in the CGPS).
    
        This function takes in a mosaic from the CGPS dataset and finds radio pairs 
        or resolved double lobed radio galaxies, and single sources in the 
        polarized intensity image. It requires functions previously defined in 
        the Functions file. It returns the information about the pairs, and also 
        creates a plot if desired. 



    Key Parameters:
        
            
            img_name (string): 
                The name of the mosaic/image you wish to go through the detections for.
                Example for the ME2 mosaic, Mo='me2'
    
            Plot_pairs (boo): 
                Whether to produce a plot of the mosaic with the pair and, if selected,
                the solo sources. Default is True to produce the plot, selecting False will skip the plot making. 
    
            
            plot_individual_sources (boo): 
                Whether to plot the solo sources found in the polarized intensity image. 
                
                Default is set to False (not plot them).
                               
            return_gal_coord (int):  
                Whether to return the twin coordinates in galactic coordinates 
                or pixel coordinates. Default is 0 for pixel coordinates 
                only, set to 1 for galactic coordinates only, and 2 for both. 
                           
                Note, the radius returned in galactic coordinates is in arcseconds.  
                If this parameter is set to 2 then it will return the pixel coordinates 
                lists first, then the galactic coordinates will follow
                for a total of 6 lists. 
                               
            mosaic_overlap (Boo): 
                Whether to remove the overlap region in the cut out areas 
                of the mosaic. Automatically set to True, which cuts out the overlap region. 
                           
    Other Parameters:
    
            PlotPI (boo): 
                Whether to plot the Polarized intensity without the detected sources. 
                          
                Default set to False. 
    
            plot_AGNs (bool): whether to plot the AGNs from Simbad.             
        
    
    
     
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
    
        

        """
    from RM_StoN import RM_code_StoN
    import parameters_file as pf
    
    
   
    
    
    
    
    
    # At some point this started loading as a string instead of a number (which 
    #   is weird since it's a number in parameters files) and this just solved that problem
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
    PIim1, StoN = fc.PIimg(Mo,  
                     plot=PlotPI, return_StoN=True)
    
    
    

        
    # Getting the polarised intensity image with the regions cut out that were 
    #   desired to be cut out
    if mosaic_overlap:    
        PIcutout, overlapcutout, maskforoverlap, maskforplot = fc.cut_out_for_mosaic(
            PIim1=PIim1, Mo =Mo, overlap=mosaic_overlap)
    else:
        PIcutout, maskforplot = fc.cut_out_for_mosaic(
            PIim1=PIim1, Mo =Mo,  overlap=mosaic_overlap)
    
    init_sources = fc.Identify_Point_Sources(PIcutout, 
                                        plot=pf.separate_individual_source_plot, threshold=pf.threshold)
    
    ### Checking if there is a corresponding source in total intensity. 
    ###     This should eliminate a lot of extended source detections. 
    # Creating an empty list to store the source info list
    initial_sources_list=[]
    TI_image = fc.T_Inten(Mo, plot=0) # Loading the total intensity image. 
    for s in init_sources: # looping through all the detected sources to get coordinates
        Y,X,r = s
        y,x = int(Y), int(X)
        PI_peak = PIim1[y,x] # finding the peak value in PI
        TI_peak = TI_image[y,x] # find the the peak value in TI
        # Checking the source has sufficient signal to noise.
        
        StoN_peak, StoN_peak_pass =RM_code_StoN(Mo,y,x, return_surrounding_pixels_above_threshold=True)
        
        if TI_peak > pf.ratio_TI_to_PI*PI_peak and StoN_peak_pass: # only adding the source if it's sufficiently large in TI and StoN
            initial_sources_list += [[y,x,r]]
    # converting the list to an array. 
    sources=np.array(initial_sources_list) 
   
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
        within_dist_range = close_enough&too_close # finding the sources that are within the range. 
        
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
       
       # Setting the number of ticks to be displayed on the plot
       tck = [n for n in range(0, len(PIim1[0]),1)]
       # Getting the labels of the x and y ticks
       tickx_labels = np.round(wx[tck],2)
       ticky_labels = np.round(wy[tck],2)
       
       
       
       # Retrieving the total intensity map. 
       TIimage = fc.T_Inten(Mo, plot=0)
       
       
       
       
       
       # Setting the cut out color maps. 
       edge_cmap= "autumn"
       overlap_cmap="Greys"
       
       # Creating the new plots. Note I need to add the coordinates from the mosaic still
       fig, ax = plt.subplots(1,2, figsize=(16,8), sharex=True, sharey=True)
       # Creating the figure title
       fig.suptitle( Mo.upper() +"\n", fontsize=1.5*pf.PTF_titlefontsize)

       # Plotting the polarized intensity map            
       Pimage = ax[0].imshow(PIim1, vmin = pf.PI_VMIN, vmax = pf.PI_VMAX,cmap=mycmap,origin='lower')
       # Creating the mask for Polarized intensity plot
       ax[0].imshow(maskforplot, alpha=pf.mask_alpha,cmap= edge_cmap, vmin=0, vmax=1,  origin='lower')
       if mosaic_overlap:
           ax[0].imshow(maskforoverlap, alpha=pf.mask_alpha,cmap= overlap_cmap, vmin=0, vmax=1,  origin='lower')
       ax[0].set_title("Polarized Intensity", fontsize=pf.PTF_titlefontsize)
       ax[0].set_xticks(tck, tickx_labels)
       ax[0].set_yticks(tck, ticky_labels)
       ax[0].set_xlabel("Longitude", fontsize=pf.PTF_axis_font_size)
       ax[0].set_ylabel("Lattitude", fontsize=pf.PTF_axis_font_size)
       fig.colorbar(Pimage, ax=ax[0], label="Jy/beam", shrink=0.74)
       ax[0].xaxis.set_major_locator(plt.MaxNLocator("auto"))
       ax[0].yaxis.set_major_locator(plt.MaxNLocator("auto"))
       
       # Plotting the total intensity map 
       
       Timage = ax[1].imshow(TIimage, vmin = pf.TI_VMIN, vmax = pf.TI_VMAX,cmap=mycmap,origin='lower')
       # Creating the mask for Polarized intensity plot
       ax[1].imshow(maskforplot, alpha=pf.mask_alpha,cmap= edge_cmap, vmin=0, vmax=1,  origin='lower')
       if mosaic_overlap:
           ax[1].imshow(maskforoverlap, alpha=pf.mask_alpha,cmap= overlap_cmap, vmin=0, vmax=1,  origin='lower')
       ax[1].set_title("Stokes I", fontsize=pf.PTF_titlefontsize)
       ax[1].set_xlabel("Longitude", fontsize=pf.PTF_axis_font_size)
       ax[1].set_ylabel("Lattitude", fontsize=pf.PTF_axis_font_size)
       ax[1].set_xticks(tck, tickx_labels)
       ax[1].set_yticks(tck, ticky_labels)
       ax[1].xaxis.set_major_locator(plt.MaxNLocator("auto"))
       ax[1].yaxis.set_major_locator(plt.MaxNLocator("auto"))
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
                                 label="Closely spaced source",
                          markerfacecolor='w', markersize=15))
       
       
    
    
    
       
       
       if plot_AGNs:
           tickx_labels_AGN = np.round(wx[tck],3)
           ticky_labels_AGN = np.round(wy[tck],3)
           

           #Getting the coordinate edges of the mosaic. 
           lmin = tickx_labels_AGN[-1]
           lmax = tickx_labels_AGN[0]
           bmin, bmax = ticky_labels_AGN[0], ticky_labels_AGN[-1]
           
           
           
           
           QSOs, dfchuck = fc.get_AGNs_CatNorth(lmin, lmax, bmin, bmax, mosaic=Mo)
           QSOs = np.round(QSOs, 3)
           
           if len(QSOs) != 0:
               # if len(QSOs)<=10:
                   # print("Gaia QSOs: \n", repr(QSOs))
               for Q in QSOs:
                   
                    l, b= Q
                    
                    x,y  = np.nanargmin(np.abs(tickx_labels_AGN-l)),np.nanargmin(np.abs(ticky_labels_AGN-b))
                    # cir = 0
                    cir1 = plt.Circle((x,y), radius = 2,  color= "gold", linewidth = 3, fill=1, alpha=0.6)
                    cir2 = plt.Circle((x,y), radius = 2,  color= "gold", linewidth = 3, fill=1, alpha=0.6)
                    ax[0].add_patch(cir1)
                    ax[1].add_patch(cir2)
               print(len(QSOs) , " Added from Gaia Catalogue")
               
               legend_elements.insert(-1, Line2D([0], [0], color='gold', lw=2, marker = 'o', markeredgewidth=5, linestyle='none',
                                          label="Quasar from Gaia",
                                   markerfacecolor='gold', markersize=15, alpha=0.6))
           DRAGNs,dfchuck = fc.get_AGNs_VLA_DRAGNs(lmin, lmax, bmin, bmax)
            
           DRAGNs = np.round(DRAGNs,3)
            
           if len(DRAGNs) != 0:
                # if len(GPQs)<=10:
                    # print("GPQs: \n", repr(GPQs))
                for D in DRAGNs:
                    
                     l, b= D
                     
                     x,y  = np.nanargmin(np.abs(tickx_labels_AGN-l)),np.nanargmin(np.abs(ticky_labels_AGN-b))
                     # cir = 0
                     cir1 = plt.Circle((x,y), radius = 2,  color= "magenta", linewidth = 3, fill=1, alpha=0.6)
                     cir2 = plt.Circle((x,y), radius = 2,  color= "magenta", linewidth = 3, fill=1, alpha=0.6)
                     ax[0].add_patch(cir1)
                     ax[1].add_patch(cir2)
                print(len(DRAGNs) , " DRAGNs from VLA")
                
                legend_elements.insert(-1, Line2D([0], [0], color='magenta', lw=2, marker = 'o', markeredgewidth=5, linestyle='none',
                                           label="DRAGN from VLA",
                                    markerfacecolor='magenta', markersize=15, alpha=0.6))
                

           Gs,dfGs= fc.get_GLADE_Gs(lmin, lmax, bmin, bmax)
            
           Gs = np.round(Gs,3)
            
           if len(Gs) != 0:
                
                for G in Gs:
                    
                     l, b= G
                     
                     x,y  = np.nanargmin(np.abs(tickx_labels_AGN-l)),np.nanargmin(np.abs(ticky_labels_AGN-b))
                     # cir = 0
                     cir1 = plt.Circle((x,y), radius = 2,  color= "purple", linewidth = 3, fill=1, alpha=0.6)
                     cir2 = plt.Circle((x,y), radius = 2,  color= "purple", linewidth = 3, fill=1, alpha=0.6)
                     ax[0].add_patch(cir1)
                     ax[1].add_patch(cir2)
                print(len(Gs) , " Gs from GLADE")
                
                legend_elements.insert(-1, Line2D([0], [0], color='purple', lw=2, marker = 'o', markeredgewidth=5, linestyle='none',
                                           label="Gs from GLADE",
                                    markerfacecolor='purple', markersize=15, alpha=0.6))
                
                
           
           # Getting the AGNs in the mosaics
           AGNs,dfchuck = fc.get_AGNs_simbad(lmin, lmax, bmin, bmax)
           AGNs = np.round(AGNs,3)
           
           # Adding the SIMBAD AGNs
           if len(AGNs)!= 0:
               
               for AGN in AGNs:
                   l, b= AGN
                   
                   x,y  = np.nanargmin(np.abs(tickx_labels_AGN-l)),np.nanargmin(np.abs(ticky_labels_AGN-b))
                   # cir = 0
                   cir1 = plt.Circle((x,y), radius = 2,  color= "darkgrey", linewidth = 3, fill=1, alpha=0.6)
                   cir2 = plt.Circle((x,y), radius = 2,  color= "darkgrey", linewidth = 3, fill=1, alpha=0.6)
                   ax[0].add_patch(cir1)
                   ax[1].add_patch(cir2)
           
               legend_elements.insert(-1, Line2D([0], [0], color='darkgrey', lw=2, marker = 'o', markeredgewidth=5, linestyle='none',
                                          label="AGN from SIMBAD",
                                   markerfacecolor='darkgrey', markersize=15, alpha=0.6))
            
           
          

       
       if plot_AGNs==False or len(AGNs)==0 and len(QSOs)==0 and len(DRAGNs)==0 :
           if plot_individual_sources:
               box_anchor= (0.5,0.88)
           else:
               box_anchor=(0.80,0.825)
           # The legend code was based off the code from the webpage: https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html 
           fig.legend(handles=legend_elements, loc="center", 
                        bbox_to_anchor=box_anchor,bbox_transform=fig.transFigure, 
                        ncol=4, fontsize='large')
       else:
           if plot_individual_sources:
               box_anchor= (0.5,0.88)
           else:
               box_anchor=(0.80,0.825)
           # The legend code was based off the code from the webpage: https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html 
           fig.legend(handles=legend_elements, loc="center", 
                        bbox_to_anchor=box_anchor,bbox_transform=fig.transFigure, 
                        ncol=4, fontsize='medium')
       
       
    
       fig.tight_layout()
       plt.show()
       plt.draw() # Allows for the axis to update when you zoom in and out
       
       from directories import pot_twin_img_dir
       plt.savefig(pot_twin_img_dir+"PDFs/" +Mo.upper()+".pdf",format="pdf")
       plt.savefig(pot_twin_img_dir+"SVGs/" +Mo.upper()+".svg",format="svg")

    if len(twinlist) ==0:
        print("There were no twins detected in mosaic "+Mo.upper())
        
        
        
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
                


        
    





def pixel_to_galactic_coordinates_axis(mosaic):
    
    """This function returns an array of the galactic coordinates of the fits image.
    
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



def offset_between_PI_and_TI_pairs(s1PI, s2PI, s1TI, s2TI, left_side, bottom):
    """This functions gives the offset between the sources detected in PI and TI.
    
    Parameters:
        s1PI(array-like): the coordinates of the first source in PI, give y then x coordinates
        
        s2PI(array-like): the coordinates of the second source in PI, give y then x coordinates
        
        s1TI(array-like): the coordinates of the first source in TI, give y then x coordinates
        
        s2TI(array-like): the coordinates of the second source in TI, give y then x coordinates
        
        left_side (int): the x pixel coordinate of the left side of the cutout in 
                         the mosaic image
        
        bottom (int): the ypixel coordinate of the bottom of the cutout in the 
                      mosaic image
        
    Returns:
       within_offset(boo): Whether or not any of the pairs were within the offset max 
       (True if within and False if offset is too large).
        """
    from parameters_file import max_offset
    # Setting the highest PI point to be s2
    if s1PI[0] < s2PI[0]:
        s1PI_y, s2PI_y = s1PI[0] - bottom, s2PI[0]- bottom
        s1PI_x, s2PI_x = s1PI[1]-left_side, s2PI[1]-left_side
    else:
        s1PI_y, s2PI_y = s2PI[0]- bottom, s1PI[0]- bottom
        s1PI_x, s2PI_x = s2PI[1]-left_side, s1PI[1]-left_side
    
    
    
    # Setting the TI point furtherest in y direction the pixel coordinates to be s2
    if int(s1TI[0]) < int(s2TI[0]):
        s1TI_y, s2TI_y = s1TI[0], s2TI[0]
        s1TI_x, s2TI_x = s1TI[1], s2TI[1]
    else:
        s1TI_y, s2TI_y = s2TI[0], s1TI[0]
        s1TI_x, s2TI_x = s2TI[1], s1TI[1]
    
    
    
   
    # Finding the offset between the x and y points in the data
    offset1x = abs(s1PI_x - s1TI_x)
    offset1y = abs(s1PI_y - s1TI_y)
    offset2x = abs(s2PI_x - s2TI_x)
    offset2y = abs(s2PI_y - s2TI_y)
    
    
    
    # Setting the within_offset value to be False if any of the offsets are greater
    #   than the max offset. 
    if (offset1x >max_offset or offset1y >max_offset or offset2y >max_offset 
        or offset2x >max_offset):
        within_offset = False
    else:
        within_offset = True
    
    # Returning whether or not the pair is within the offset threshold/tolarance. 
    return within_offset


def num_sources_in_correlation_regions(s1PI, s2PI, TISources, left_side, bottom):
    """ This function finds the closest Stokes I sources to the twins found in polarised
        intensity. Design for when 3 or more sources in Stokes I are detected, but
        could function for any number unless none were detected. 
        
        Parameters:
            s1PI (array): The coordinates of the first twin from polarised intensity, (y,x)\
                
            s2PI (array): The coordinates of the first twin from polarised intensity, (y,x)
            
            TIsources (array): An array of the coordinates of the sources detected in 
                Stokes I cut out. 
                
            left_side (int): The pixel coordinate of left side of where the cut 
                out starts in full mosaic
                
            bottom (int): The coordinates of where the bottom of the cut out in
                the mosaic.
            
        Returns:
            num_sources_in_cor_regions (int): The number of sources in the correlation 
                region of the twins."""
    
    from parameters_file import max_offset
    
    max_offset_radius = max_offset #np.sqrt(2*(max_offset**2))
    # getting the coordinates for the polarized intensity pairs cut out in the  
    #  TI cutout
    p1 = s1PI[0] - bottom, s1PI[1]-left_side # p = x, y 
    p2 = s2PI[0] - bottom, s2PI[1]-left_side # p = x, y 
    
    # Finding the difference between the x and y coordinates for all the sources 
    #   detected in TI and the 1st twin in polarized intensity 
    dy_p1 = abs(TISources[:,0] - p1[0])
    dx_p1 = abs(TISources[:,1] - p1[1])
    
    #Doing the same but for the second twin
    dy_p2 = abs(TISources[:,0] - p2[0])
    dx_p2 = abs(TISources[:,1] - p2[1])
    
    
    # Finding the distance between each TI source and the twins
    dist_p1 = np.sqrt((dy_p1**2)+(dx_p1**2))
    dist_p2 = np.sqrt((dy_p2**2)+(dx_p2**2))
    
    
    # Determinng which sources are in the correlation region of each twin
    in_cor_region_1 = dist_p1 <= max_offset_radius
    in_cor_region_2 = dist_p2 <= max_offset_radius
    
    # Prevents double counting if one source is both regions
    combined_cor_regions = (in_cor_region_1 + in_cor_region_2)> 0
    
    num_sources_in_cor_regions = np.sum(combined_cor_regions)
    
    
    return num_sources_in_cor_regions
    

def closest_TI_sources_to_PI_pair(s1PI, s2PI, TISources, left_side, bottom):
    """ This function finds the closest Stokes I sources to the twins found in polarised
        intensity. Design for when 3 or more sources in Stokes I are detected, but
        could function for any number unless none were detected. 
        
        Parameters:
            s1PI (array): The coordinates of the first twin from polarised intensity, (y,x)\
                
            s2PI (array): The coordinates of the first twin from polarised intensity, (y,x)
            
            TIsources (array): An array of the coordinates of the sources detected in 
                Stokes I cut out. 
                
            left_side (int): The pixel coordinate of left side of where the cut 
                out starts in full mosaic
                
            bottom (int): The coordinates of where the bottom of the cut out in
                the mosaic.
                
            
        Returns:
            TI1_index (int or nan): The index of the source in TI that is closest to s1. 
                If there were no sources in the correlation it will return a nan.
                
            TI2_index (int or nan): The index of the source in TI that is closest to s2,
                if it's not the same as TI1_index. If it is the same, or if there 
                were no sources in the correlation it will return a nan.
            
        """
    
    from parameters_file import max_offset
    # getting the coordinates for the polarized intensity pairs cut out in the  
    #  TI cutout
    
    p1 = s1PI[0] - bottom, s1PI[1]-left_side # p = x, y 
    p2 = s2PI[0] - bottom, s2PI[1]-left_side # p = x, y 
    
    # Finding the difference between the x and y coordinates for all the sources 
    #   detected in TI and the 1st twin in polarized intensity 
    dy_p1 = abs(TISources[:,0] - p1[0])
    dx_p1 = abs(TISources[:,1] - p1[1])
    
    #Doing the same but for the second twin
    dy_p2 = abs(TISources[:,0] - p2[0])
    dx_p2 = abs(TISources[:,1] - p2[1])
    
    
    # Finding the distance between each TI source and the twins
    dist_p1 = np.sqrt((dy_p1**2)+(dx_p1**2))
    dist_p2 = np.sqrt((dy_p2**2)+(dx_p2**2))
    
    
   
    
    # Determining which source is closest to the pairs detected in total intensity
    TI1_index = np.argmin(dist_p1)
    TI2_index = np.argmin(dist_p2)
    
    
    
    # Checking if the sources are in the correlation region. 
    TI1_in_offset = dx_p1[TI1_index] <=max_offset and dy_p1[TI1_index] <=max_offset
    TI2_in_offset = dy_p2[TI2_index] <=max_offset and dx_p2[TI2_index] <=max_offset
    
    
    
        # Could change the code to see test if it's in the offset first. 
    # Checking if there is one source that is closest to both PI twins
    if TI1_index != TI2_index:
        
        # returning the index of both sources if they are both in the offset region
        if TI1_in_offset and TI2_in_offset:
            return TI1_index,TI2_index
        
        # returning the index of one source if only one is in the offset region
        elif TI1_in_offset and TI2_in_offset==False:
            return TI1_index, np.nan
        
        # returning the index of one source if only one is in the offset region
        elif TI1_in_offset==False and TI2_in_offset:
            return TI2_index, np.nan
        
        # returning nans if neither are in the correlation region. 
        else:
            return np.nan, np.nan
    
    # Checking if the source is in the correlation
    elif TI1_in_offset:
        # Creating a new source list with the closest source removed to
        #   to find the second closest source
        other_sources = np.delete(TISources, TI1_index, axis=0)
        
        # Finding the difference between the x and y coordinates for 
        #   all the sources detected but the closest source in TI and
        #   the 1st twin in polarized intensity 
        dy_p1_check = abs(other_sources[:,0] - p1[0])
        dx_p1_check = abs(other_sources[:,1] - p1[1])
        
        # Same thing but for the 2nd twin
        dy_p2_check = abs(other_sources[:,0] - p2[0])
        dx_p2_check = abs(other_sources[:,1] - p2[1])
        
        
        # Finding the distance sources
        dist_p1_check = np.sqrt((dy_p1_check)**2+(dx_p1_check)**2)
        dist_p2_check = np.sqrt((dy_p2_check)**2+(dx_p2_check)**2)
        
        # Finding the index of the closest source
        possible_2nd_source1_index = np.argmin(dist_p1_check)
        possible_2nd_source2_index = np.argmin(dist_p2_check)
        
        # If the smallest distance to twin 1 is smaller than the smallest
        #   distance to twin 2. 
        if (dist_p1_check[possible_2nd_source1_index]
            <dist_p2_check[possible_2nd_source2_index]):
            
            # If the source is stil within the offset/correlation region:
            if (dy_p1_check[possible_2nd_source1_index]<=max_offset 
                and dx_p1_check[possible_2nd_source1_index]<=max_offset):
                
                # Set the second index to be the second closest source.
                TI2_index=possible_2nd_source1_index
                return TI1_index, TI2_index
            
            # The next closest source is not in the correlation region so only return 1 value
            else:
            
                return TI1_index, np.nan
        else:# if the smallest distance to twin 2 is smaller than the smallest distance to twin 1.
        
            # If the source is stil within the offset/correlation region:
            if (dy_p2_check[possible_2nd_source2_index]<=max_offset 
                and dx_p2_check[possible_2nd_source2_index]<=max_offset):
                
                # Set the second index to be the second closest source.
                TI2_index=possible_2nd_source2_index
                return TI1_index, TI2_index
            
            # The next closest source is not in the correlation region so only return 1 value
            else:
            
                return TI1_index, np.nan
            
    # Checking if the source is in the correlation
    elif TI2_in_offset:
        # Creating a new source list with the closest source removed to
        #   to find the second closest source
        other_sources = np.delete(TISources, TI2_index, axis=0)
        
        # Finding the difference between the x and y coordinates for 
        #   all the sources detected but the closest source in TI and
        #   the 1st twin in polarized intensity 
        dy_p1_check = abs(other_sources[:,0] - p1[0])
        dx_p1_check = abs(other_sources[:,1] - p1[1])
        
        # Same thing but for the 2nd twin
        dy_p2_check = abs(other_sources[:,0] - p2[0])
        dx_p2_check = abs(other_sources[:,1] - p2[1])
        
        
        # Finding the distance sources
        dist_p1_check = np.sqrt((dy_p1_check)**2+(dx_p1_check)**2)
        dist_p2_check = np.sqrt((dy_p2_check)**2+(dx_p2_check)**2)
        
        # Finding the index of the closest source
        possible_2nd_source1_index = np.argmin(dist_p1_check)
        possible_2nd_source2_index = np.argmin(dist_p2_check)
        
        # If the smallest distance to twin 1 is smaller than the smallest
        #   distance to twin 2. 
        
        if (dist_p1_check[possible_2nd_source1_index]
            <dist_p2_check[possible_2nd_source2_index]):
            
            # If the source is stil within the offset/correlation region:
            if (dy_p1_check[possible_2nd_source1_index]<=max_offset 
                and dx_p1_check[possible_2nd_source1_index]<=max_offset):
                
                # Set the second index to be the second closest source.
                TI2_index=possible_2nd_source1_index
                return TI1_index, TI2_index
            
            # The next closest source is not in the correlation region so only return 1 value
            else:
            
                return TI1_index, np.nan
        else:# if the smallest distance to twin 2 is smaller than the smallest distance to twin 1.
        
            # If the source is stil within the offset/correlation region:
            if (dy_p2_check[possible_2nd_source2_index]<=max_offset 
                and dx_p2_check[possible_2nd_source2_index]<=max_offset):
                
                # Set the second index to be the second closest source.
                TI2_index=possible_2nd_source2_index
                return TI1_index, TI2_index
            
            # The next closest source is not in the correlation region so only return 1 value
            else:
            
                return TI1_index, np.nan
       
    
    # None of the sources are in the correlation region
    else:
        return np.nan, np.nan
        
            
            
    
    
def solo_offset_test(s, center_x, center_y, return_offset=False):
    
    """This function determines if the single source detected in TI is within 
    the correlation region of the twin pairs. This region is located around the 
    center of the source. 
    
    Parameters:
        s (arraylike): an array that contains the coordinates, HWHM, and peak of the solo source
        
        center_x (int): the x pixel coordinate of the center of the twins in PI
        
        center_y (int): the y pixel coordinate of the center of the twins in PI
        
        return_offset (Bool, optional): whether to return the x and y offset. Default is False
        
    Returns:
        
        within_offset (Bool): returns True if the source in is in the region, returns False otherwise
        
        if return_offset is True it also returns:
            offset_x (int): the offset in pixel coordinates in the x direction
            
            offset_y (int): the offset in pixel coordinates in the y direction
        
    """
    
    from parameters_file import max_offset
    
    # Getting the individual coordinates
    TI_y, TI_x, TI_r,TI_p = s 
    
    # Finding the offset between the center of the PI pair and the source found in TI
    offset_x = abs(TI_x - center_x)
    offset_y = abs(TI_y - center_y)
    
    offset_dist = np.sqrt((offset_x)**2 +(offset_y)**2)
    
    # Classifying the source based on the offset.
    if offset_dist<=max_offset:# and offset_y<= max_offset:
        within_offset=True
    else:
        within_offset=False
        
    
    
    if return_offset:
        return within_offset, offset_x, offset_y
    else:
        return within_offset

def print_classifications():
    print("The Classifications of the pairs are as follows:")
    print("None: No sources were detected in Stokes I")
    print("0.50: A single source was detected in Stokes I and is NOT correlated to \n \t the "\
          + "pair of sources found in polarised intensity")
    print("1.00: A single source was detected in Stokes I and IS correlated to \n \t the "\
          + "pair of sources found in polarised intensity")
    print("2.00: two close sources (within 15 pixels) were are a twin")
    print("2.25: two close sources (within 15 pixels) were detected but BOTH were"\
          + " OUTSIDE the offset tolerance")
    print("2.50: two close sources (within 15 pixels) were detected but ONLY ONE"\
          +" source was within the offset tolerance.")
    print("3.00: two sources far apart (>15 pixels) were detected within the cutout "\
          +"but ONLY ONE correlated to a source")
    print("3.25 two sources far apart (>15 pixels) were detected both sources "\
          +"correlated. Note: I don't think this will occur, but just in case. ")
    print("3.50: two sources far apart (>15 pixels) were detected within the "\
          +"cutout but both were too fair apart to be twins")
    print("4.00: Twin found and no other sources correlate to the pair found in Stokes I (or total intensity)"\
            +"\n4.25: Twin found with another or multiple other sources are correlated with it in Stokes I (or total intensity)"\
            +"\n4.50: No twin was found but a single correlated source was found "\
            +"\n4.75: no twins were found but multiple correlated sources were found. "\
            +"5.00: No correlated sources were found despite detecting more than 3 sources in Stokes I/total intensity. ")
    
        

def solo_source_central(s, cx,cy, PIs1, PIs2, left_side, bottom):
    """This function checks if the solo source is closer to the center than it is
    to the two sources found in PI. If the one source TI is correlated to both sources,
    it should be closer to the center. 
    
    Inputs: 
        s (array): SI Source coodinates
        
        cx (int) : x pixel coordinate of the pair in the cutout
        
        cy (int) : y pixel coordinate of the pair in the cutout
        
        PIs1 (array): pixel coordinates of the first PI source in the mosaic
        
        PIs2 (array): pixel coordinates of the second PI source in the mosaic
        
        left_side (int): the x pixel coordinate of the left side of the cutout in 
                         the mosaic image
        
        bottom (int): the ypixel coordinate of the bottom of the cutout in the 
                      mosaic image
                      
    Returns:
        Boolean value, whether or not a the single SI source is closer to one PI 
        or the center between both sources
    
    """
    
    sx, sy, sr, sp = s
    dist_central = np.sqrt((sx-cx)**2+(sy-cy)**2)
    
    s1y, s1x, s1r, s1p = PIs1
    s2y, s2x, s2r, s2p = PIs2
    
    
    s1y, s2y, s1x, s2x = s1y-bottom,s2y-bottom,s1x-left_side,s2x-left_side
    
    dist_s1 = np.sqrt((s1y-sy)**2+(s1x-sx)**2)
    dist_s2 = np.sqrt((s2y-sy)**2+(s2x-sx)**2)
    
    
    if dist_central < dist_s1 and dist_central < dist_s2:
        return True
    else:
        return False
    
    
    

    

def TI_twin_detector_and_binary_pair_classifiers(TI_cutout_sources, c, twinPI, left_side,
                                          bottom, return_sibling_sources=False):
    """ This function detects twins and classifies them based on number of sources
        in the TI cutout, the distance between them, and offset between them and PI.
        
    Parameters:
        TI_cutout_sources (array): the sources detected in Total intensity.
        
        c (array): the coordinates (y,x) of the center of the twin in total intensity.
        
        twinPI (array): the coordinates of the twin pair in polarised intensity, 
                        [[y1,x1],[y2,x2]].
                        
        left_side (int): the x pixel coordinate of the left side of the cutout in the 
                         mosaic image
        
        bottom (int): the ypixel coordinate of the bottom of the cutout in the 
                         mosaic image
                    
        return_sibling_sources (Boo): Whether to return the sibling source list or not.
            Default is False. 
            
    Returns: pair_classifications, TI_twins, all_twin_sources, solo_sources, 
        pair_classifications: a list containing the classification of each cutout
        
        TI_twins: A list with the coordinates of the sources deemed a twin.
                    Of the form:[[[y1,x1,r1], [y2,x2,r2]], ...]
                    
        all_twin_sources:  a list containing all the individual twin sources.
        
        solo_sources: a list of sources detected in the image that are not a twin pair.
        
        central_sources: a list of sources correlated to the center of the PI twin pair. 
        
        sibling_sources (if selected): the sources that are in correlation region, but are too far apart to be twins. 
        
        """
    
    # from parameters_file import max_offset
    from parameters_file import max_dist_btw_sources as max_dist
    from parameters_file import ratio_threshold_PI, ratio_threshold_TI
    # max_dist = max_dist_btw_sources
    
    twin_detector =True
    
    # Getting the coordinates of the twins in PI
    s1PI, s2PI = twinPI
    s1PIy, s1PIx,s1PIr,s1PIp = s1PI
    s2PIy, s2PIx,s2PIr,s2PIp = s2PI
    
    
   
        
    # Finding the center of the polarised intensity pair. The coordinates need to be in the 
    c_y, c_x =c
    center_y, center_x= round(c_y)-bottom, round(c_x)-left_side
    
    # Creating lists to store the source information    
        # A list containing the twins found in total intensity.
    TI_twins =[]
        # a list to store the individual twin sources, this is to prevent double counting pairs
    all_twin_sources = []
        
    
    # Creating a list to store solo sources found in TI
    solo_sources=[]
    
    # A double sources list of sources
    sibling_sources=[]
    
    central_sources=[]# a list of sources correlated to the center of the PI twins. 
    # Creating a list to store the discarded but detected sources. 
    # discarded_sources=[]
    
    
    ##################### Pair Classifications #####################
    # Starting with a blank binary classification  
    binary_classification =0
    
    # Checking if only one source in the list.
    if len(TI_cutout_sources)==1:
        # Getting the coordinates of the source found
        s = TI_cutout_sources[0]
        
        binary_classification =2**0 # Flag for 1 source detected
        
        # Finding the offset between the center of the PI pair and the source found in TI
        within_offset = solo_offset_test(s, center_x=center_x, center_y=center_y)
        
        
        
        # Classifying the source based on the offset.
        if within_offset:
            
            solo_sources.append(s.tolist())
            
            central_source = solo_source_central(s=s,cx=center_x, cy=center_y, 
                                                 PIs1=s1PI, PIs2=s2PI, left_side=left_side, bottom=bottom)
            if central_source:
                central_sources.append(s.tolist())
                binary_classification += 2**4 # Flag for one source in the cor. region
            else:
                binary_classification += 2**3 # Flag for no sources in the cor. region of center
        else:
            # discarded_sources.append(s) -> decided to not keep track of these since doing that would hard when there are 3+ sources in the cutout
            binary_classification += 2**3 # Flag for no sources in the cor. region of center
        
        
        
    # Setting the Classification if there are no sources detected.   
    elif len(TI_cutout_sources)==0:

        binary_classification =None # Flag for no sources detected
    
    
    
    
    # Going through the classification procedure if there is only two sources are found.
    elif len(TI_cutout_sources)==2:
        
        # if 
        binary_classification = 2**1 # Flag for two sources detected
        # Setting the num of sources in the correlation region to 0 initially. 
        num_of_correlated_solo_sources=0 
        num_of_central_solo_sources=0
        
        
        # Getting the coordinates of the sources
        # s1_tempt, s2_tempt = TI_cutout_sources[0],TI_cutout_sources[1]
        # s1, s2 = s1_tempt.tolist(), s2_tempt.tolist()
        
        s1, s2 = TI_cutout_sources[0].tolist(), TI_cutout_sources[1].tolist()
        s1y, s1x, s1r, s1p = s1
        s2y, s2x, s2r, s2p = s2
        
        # Finding the ratio between the two peaks
        if s1p >s2p:
            outside_TI_ratio=s1p/s2p >= ratio_threshold_TI
        else:
            outside_TI_ratio=s2p/s1p >= ratio_threshold_TI
       
        
        # Finding the distance in the x and y between the two TI sources
        dist_x = abs(s1x-s2x)
        dist_y = abs(s1y-s2y)
        
        # Finding the diagnoal distance between the two sources 
        dist_xy = np.sqrt((dist_x**2)+(dist_y**2))
        
        
        
        # Checking if the distance between the two sources is within the maximum 
        #   distance for them to be considered a twin pair.
        if dist_xy <max_dist:
            
            # Checking if the sources are within the offset. 
            within_offset = offset_between_PI_and_TI_pairs(s1PI, s2PI, 
                                                           s1, s2, left_side, bottom)
            
            if within_offset:
                
                
                binary_classification += 2**5 #Flag for two sources in the correlation region.
                if outside_TI_ratio ==False:
                    binary_classification += 2**9 *(twin_detector) # Flag for twin detected. 
                    # Classifying and adding the twins to the lists 
                    TI_twins.append([s1,s2])
                    all_twin_sources.append(s1)
                    all_twin_sources.append(s2)
                else:
                    binary_classification += (2**8) *(outside_TI_ratio) # Flag for ratio between peaks to large. 
                    
                
            else: # if it's not in the offset region
                
                for S in TI_cutout_sources.tolist():
                    # Checking if any of the sources is correlated to the center of PI pair. 
                    within_offset = solo_offset_test(S, center_x=center_x, center_y=center_y)
                    
                    # Checking if the source is within the offset.
                    if within_offset:
                        central_source = solo_source_central(s=S,cx=center_x, cy=center_y, 
                                                             PIs1=s1PI, PIs2=s2PI, left_side=left_side, bottom=bottom)
                        if central_source:
                            num_of_central_solo_sources +=1 # Flag for no sources in the cor. region
                         
                        num_of_correlated_solo_sources += 1
                        solo_sources.append(S)
                    
                # Classifying the sources 
                if num_of_correlated_solo_sources==0:
                   binary_classification += 2**3 # Flag for no sources in cor. region
                elif num_of_correlated_solo_sources==2:
                   binary_classification += 2**5# Flag for two sources in cor. region
                elif num_of_correlated_solo_sources==1 and num_of_central_solo_sources==1:
                    binary_classification += 2**4 # Flag for one source in cor. region
                    central_sources.append(solo_sources[0])
                else:
                    binary_classification += 2**3 # Flag for no sources in central cor. region
                
         
        else:  # if the distance between detected sources is too great 
            num_of_central_solo_sources=0
            # binary_classification += 2**7  # if the distance between detected sources is too great      
            
            num_sources_in_cor_regions =num_sources_in_correlation_regions(s1PI,
                                        s2PI, TI_cutout_sources, left_side, bottom)
            
            TI1_index, TI2_index =closest_TI_sources_to_PI_pair(s1PI, s2PI, TI_cutout_sources,
                                                                left_side, bottom)
            
            #TI2_index returns np.nan there is only one or no sources in the cor region
            both_cor = TI2_index != np.nan 
            
            if num_sources_in_cor_regions==2:
                binary_classification +=2**5 +2**7
            elif num_sources_in_cor_regions==0:
                binary_classification +=2**3
            else:# num_sources_in_cor_regions=1
                central_source= solo_source_central(TI_cutout_sources[int(TI1_index)],
                                                    cx=center_x, cy=center_y, PIs1=s1PI,
                                                    PIs2=s2PI, left_side=left_side, bottom=bottom)
                
                if central_source:
                    binary_classification+=2**4
                else:
                    binary_classification+=2**3
            
            
            
            
            
            # if num_sources_in_cor_regions !=0:
            for S in TI_cutout_sources.tolist():
                central_source = solo_source_central(s=S,cx=center_x, cy=center_y, 
                                                     PIs1=s1PI, PIs2=s2PI, left_side=left_side, bottom=bottom)
                num_of_central_solo_sources += central_source
                   
                 
                # Checking if the sources are within the central cor region
                within_offset = solo_offset_test(S, center_x=center_x, center_y=center_y)
                if within_offset and num_of_correlated_solo_sources ==0:

                    num_of_correlated_solo_sources += 1
                    solo_sources.append(S)
                    
                elif within_offset and num_of_correlated_solo_sources == 1:
                    solo_sources.append(S)
                    num_of_correlated_solo_sources += 1
                    print("Two sources correlated to the center but is not a twin, weird")
            
            
            # If there are two sources in the cor region but neither is correlated to both/the center.
            #   I don't actually think this will happen but I wanted a way to keep track of it if I did.
            if num_of_correlated_solo_sources==0 and both_cor: 
                sibling_sources.append(TI_cutout_sources[0].tolist())
                sibling_sources.append(TI_cutout_sources[1].tolist())

            elif num_of_correlated_solo_sources ==1 and num_of_central_solo_sources==1:
                central_sources.append(solo_sources[0])
                
           
                
     
       
    else: # if there were more than 2 sources detected:
        binary_classification = 2**2 # Flag for 3 or more sources detected.
        
        
        
        # Getting the coordinates of the twin pair in PI
        s1PI, s2PI = twinPI 
        # Getting the indices of the sources that are closest to the PI pair
        TI1_index, TI2_index =closest_TI_sources_to_PI_pair(s1PI, s2PI, TI_cutout_sources,
                                                            left_side, bottom)
        
         
        # Calculating the number of sources in the cor. region of each twin
        num_sources_in_cor_regions =num_sources_in_correlation_regions(s1PI,
                                    s2PI, TI_cutout_sources, left_side, bottom)
        
        
        if num_sources_in_cor_regions ==2:
            
            binary_classification+=2**5 # Flag for two sources in the cor region.
            
            
            # Getting the coordinates for the sources
            y1,x1,r1,p1 = TI_cutout_sources[int(TI1_index)]
            y2,x2,r2,p2 = TI_cutout_sources[int(TI2_index)]
            
            if p1>p2:
                outside_TI_ratio=p1/p2>=ratio_threshold_TI
            else:
                outside_TI_ratio=p2/p1>=ratio_threshold_TI
            
            
            distance = np.sqrt((y1-y2)**2 +(x1-x2)**2)
            
            if distance<=max_dist:
                if outside_TI_ratio:
                    # Adding a flag for peaks being too different
                    binary_classification += (2**8) *(outside_TI_ratio)
                else:
                    binary_classification+=(2**9)*(twin_detector) # Flag for twin detection
                
                    TI_twins.append([TI_cutout_sources[TI1_index].tolist(),
                                     TI_cutout_sources[TI2_index].tolist()])
                    
                    all_twin_sources.append(TI_cutout_sources[TI1_index].tolist())
                    all_twin_sources.append(TI_cutout_sources[TI2_index].tolist())
            else:
                binary_classification += 2**7 # Flag sources being too fair apart to be twins
                # sibling sources are sources in the cor region but aren't twins or solo sources.
                sibling_sources.append(TI_cutout_sources[TI1_index].tolist())
                sibling_sources.append(TI_cutout_sources[TI2_index].tolist())
                
            
        elif num_sources_in_cor_regions==1:
            s = TI_cutout_sources[int(TI1_index)]
            # Check if the source in total intensity if correlated with the center of the source.
            within_offset_solo = solo_offset_test(s, center_x=center_x, center_y=center_y)

            central_source = solo_source_central(s, cx=center_x, cy=center_y, PIs1=s1PI, 
                                                 PIs2=s2PI, left_side=left_side, bottom=bottom)

            if within_offset_solo and central_source:
                binary_classification += 2**4 # Flag for 1 source in central cor region
                solo_sources.append(TI_cutout_sources[TI1_index].tolist())
                central_sources.append(TI_cutout_sources[TI1_index].tolist())
            else:
                binary_classification += 2**3 # Flag for 0 sources in central cor region
            
            

    
        elif num_sources_in_cor_regions ==0:
            binary_classification +=2**3  # Flag for 0 sources in central cor region
        
         
            
        elif num_sources_in_cor_regions <5: # 3 to 4 sources in the correlation region 
            binary_classification +=2**6 # Flag for when there are 3+ sources in cor region
            
            # getting coordinates of the closest sources 
            y1,x1,r1, p1 = TI_cutout_sources[TI1_index] 
            y2,x2,r2,p2 = TI_cutout_sources[TI2_index]
            
            if p1>p2:
                outside_TI_ratio=p1/p2>=ratio_threshold_TI
            else:
                outside_TI_ratio=p2/p1>=ratio_threshold_TI
                
            
            distance = np.sqrt((y1-y2)**2 +(x1-x2)**2)
            
            if distance<=max_dist:
                
                if outside_TI_ratio==True:
                    # Adding a flag for peaks being too different
                    binary_classification += (2**8) *(outside_TI_ratio) 
                else:
                    binary_classification+=(2**9)*(twin_detector) # Flag for twin detection
                # binary_classification+=(2**9)*(twin_detector) # Flag for twin detection
                
                    # Adding the sources to the lists
                    TI_twins.append([TI_cutout_sources[TI1_index].tolist(),
                                     TI_cutout_sources[TI2_index].tolist()])
                    
                    all_twin_sources.append(TI_cutout_sources[TI1_index].tolist())
                    all_twin_sources.append(TI_cutout_sources[TI2_index].tolist())
            else:
                binary_classification += 2**7 # Flag sources being too far apart to be twins
                # sibling sources are sources in the cor region but aren't twins or solo sources.
                sibling_sources.append(TI_cutout_sources[TI1_index].tolist())
                sibling_sources.append(TI_cutout_sources[TI2_index].tolist())
            
            # Checking, which indices is greater to know what order to delete the sources in
            if TI1_index<TI2_index:
                n1, n2 = TI1_index,TI2_index
            else:
                n1, n2 = TI2_index,TI1_index
                
            if num_sources_in_cor_regions ==len(TI_cutout_sources):
                # Creating a list of sources without the two closest sources in it 
                other_sources= np.delete(np.delete(TI_cutout_sources, n2, axis=0), n1, axis=0) 
                for o in other_sources:
                    # Note: TI3_index should not be np.nan if there are more than 3 sources in the correlation region so we don't need to test it.
                    sibling_sources.append(o.tolist())
            else:
                # Creating a list of sources without the two closest sources in it 
                other_sources= np.delete(np.delete(TI_cutout_sources, n2, axis=0), n1, axis=0) 
                
                
                # Getting the other sources in the cor region.
                TI3_index, TI4_index =closest_TI_sources_to_PI_pair(s1PI, s2PI, other_sources,
                                                                    left_side, bottom)
                
                # Note: TI3_index should not be np.nan if there are more than 3 sources in the correlation region so we don't need to test it.
                sibling_sources.append(TI_cutout_sources[TI3_index].tolist())
                # Checking if there is another to add to the siblings list.
                
                
                if ~np.isnan(TI4_index):
                    sibling_sources.append(TI_cutout_sources[TI4_index].tolist())
            
        
        
        # I really do not expect this to happen, but if it does I will update the code. 
        else:
            print("Huston we have a problem, more than 4 correlated sources.")   
                
        
    if return_sibling_sources:                    
        return binary_classification, TI_twins, all_twin_sources, solo_sources, \
            central_sources, sibling_sources,
    else:
        return binary_classification, TI_twins, all_twin_sources, solo_sources, central_sources

def binary_classification_dictionaries():
    
    
    dictionary_words = {None: "No sources were detected in the cutout.",
                        
                  17: "One solo source in Stokes I cutout was correlated with the two in polarised intensity.",
                  9: "One source was detected in the Stokes I cutout but was not correlated with the sources in Polarized intensity.",
                  
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
                  False: "A false detection occured.",
                  "F": "A false detection occured.",
                  "f": "A false detection occured.",
                  256:"The ratio between the PI is too large.",
                  290:"Two sources were detected in Stokes I but the ratio between the peaks in TI is too large",
                  324: "Multiple sources were detected in Stokes I, but the two closest sources TI peak ratios were too large",
                  292:"Two sources were in the correlation region but the TI peak ratio was too large."
                  }
    # list = [number of detected sources, sources within the cor region, twin detected, sources close enough together, peaks within threshold]
    dictionary_key = {None:[None, None, False, False],
                      17: [1, 1, False, False],
                      9:  [1,0,False, False],
                      546: [2,2, True,False],
                      18: [2, 1, False,False],
                      10: [2,0, False, False],
                      146: [2, 1, False, False],
                      162: [2, 2, False, False],
                      138: [2,0, False, False],
                      12: [3, 0, False, False],
                      20: [3, 1, False, False],
                      548:[3, 2,True, False],
                      164: [3, 2, False, False],
                      196: [3, 3, False, False],
                      580: [3, 3, True, False],
                      False: [None, None, None, False],
                      "F": [None, None, None, False],
                      "f": [None, None, None, False],
                      256: [None, None, None, True],
                      290: [2,2, True,True],
                      292:[3, 2,True, True],
                      324: [3, 3, True, True],
                      }
    
    return dictionary_words, dictionary_key



        
        
        
        
            
    
def twin_total_intensity_detector_and_classifier(mosaic, centers, twin_list, 
                                                 distance,
                                                 plot_snapshots=True,
                                                 return_singular_list=True, 
                                                 plot_AGNs=True, 
                                                 ):
    
    """
This function classifies the galatic potnetial twin sources by seeing if
there is a matching set of sources in Stokes I.
        
Parameters:
    
    mosaic (str): 
        the mosaic the set of twins is from. 
    
    centers (list): 
        a list of the center of the twin pairs.
    
    twin_list (list): 
        a list of twin pairs containing the pixel coordinates 
        for each source
     
    distance (list):
        a list of the number the pixel distance between the potential twins
          
    plot_snapshots (Boo): 
        Whether plot the snapshots or not.
        
        Default is True
        
    return_data_list(Boo): 
        Whether to return the data for the twins in one list or separate ones. 
    
    plot_AGNs(bool): whether to plot the AGNs found in other surveys. Currently 
        the catalogues consist of all SIMBAD AGN sources excluding some Blazars, 
        the CATNORTH GAIA quasar catalogue, and the VLA DRAGNs survey. 
         
        
Returns:
                
    mosaic_TI_twins (list): 
        a list of the twins detected in Stokes I.
        
        List has the form: [ [[y,x,r], [y,x,r]], [[y,x,r], [y,x,r]], ...]
        
    mosaic_solo_sources (list): 
        A list of the single sources that were 
        correlated to the center of the pair found in polarised intensity.
        
    mosaic_true_classifications (list): 
        a list of the true classifications 
        of the pairs.
    
    detected_sources_list (list): 
        A list of sources found in Stokes I using 
        the point source detection algorithm. 
        
    all_mosaic_twin_sources (list): 
        A list of all the twins detected in 
        the mosaic. Same list as mosaic_TI_twins but without a dimension 
        pairing the twins together. 
        
        List has the form: [[y,x,r], [y,x,r], ...]
    
        """
    #making image directories
    
    from directories import cutout_img_dir
    from os import makedirs
    
    
    # Creating the folder name. The RM code requires the name not include the 
    #   "m" at the start of the folder name. So "mf3" needs to be named "f3".
    #   The RM code also requires the dat file for each mosaic to be it's own 
    #   separate folder.
    if mosaic[0] != 'm' and mosaic[0] != "M":
        foldername = mosaic.lower()
    else:
        foldername = mosaic[1:].lower()
        
    # Setting the output directory of the dat file. 
    out_dir = f'{cutout_img_dir}{foldername}/'
    
    # Making a new folder 
    makedirs(out_dir, exist_ok=True)
    
    import parameters_file as pf 
    
    
   
    
    
    # This threshold is 10x the threshold for polarized intensity because only 
    #   a fraction of the light received is polarized, so there will
    #   be much more light detected in total intensity than in PI. 
    
    
    
    
    plot_detection_circles=True
    
    # radius_scale =2
    
    initial_graph_time=pf.initial_graph_time#(seconds)
    
    
    snapshot_full_length = pf.snapshot_length
    snapshot_width = int(snapshot_full_length/2)
    
    
    
    ####### Creating empty data file to store all the info in ######
    
    twin_dataset = []
    
    
    ######## Code begins#########
    # Creating an array to store all the cutout images
    
    
    
    if len(twin_list)==0:
        
        return None
    
   

    # Getting the total intensity image of the mosaic
    TI_mosaic= fc.T_Inten(mosaic, plot=0)
    
    PI_mosaic, StoN = fc.PIimg(mosaic, plot=0,return_StoN = True)
    
    
    
    fontsize=15
    # Finding the height and width of the mosaic
    TI_width=len(TI_mosaic[0])
    TI_height=len(TI_mosaic[0:,])
    
    #initializing lists to store the center of the twins in 
    centers_long, centers_lat, distances_arcmin =[],[],[]
    
    # Going through every pair of twins
    for n,c in enumerate(centers):
        
        
        print("\n PI twin pair: ", n)
        # Getting coordinates and peaks
        twin1PI, twin2PI = twin_list[n]
        t1PI_y, t1PI_x, t1PI_r, t1PI_p = twin1PI
        t2PI_y, t2PI_x, t2PI_r, t2PI_p = twin2PI
        
        min_PI_peak = min(t1PI_p, t2PI_p)
        
        # Finding Signal to Noise of the twin peaks.
        # t1_StoN, t2_StoN = StoN[int(t1PI_y), int(t1PI_x)], StoN[int(t2PI_y), int(t2PI_x)]
        t1_StoN, t2_StoN = RM_code_StoN(mosaic, int(t1PI_y), int(t1PI_x)), RM_code_StoN(mosaic, int(t2PI_y), int(t2PI_x))
        
        print("L1970 StoN Twin 1, Twin 2: ", t1_StoN, t2_StoN )
        
        # Getting the coordinates of the center of the twins to make the cutouts.
        #   Note: this uses pixel coordiantes, not galactic coordinates
        c_y, c_x =c
        center_y, center_x= round(c_y), round(c_x)
       
        # Finding the value of the left side of the cutout
        if center_y>snapshot_width:
            bottom = center_y -snapshot_width
        else: 
            bottom=0
        
        # Finding the value of the right side of the cutout
        if center_y< TI_height-snapshot_width:
            top=center_y+snapshot_width
        else:
            top=TI_width
        
        # Finding the bottom value of the cutout
        if center_x>snapshot_width:
            left_side = center_x -snapshot_width
        else: 
            left_side=0
            
        # Finding the top value of the cutout
        if center_x< TI_height-snapshot_width:
            right_side=center_x+snapshot_width
        else:
            right_side=TI_width
        
        
        y_center_TI, x_center_TI  = center_y - bottom, center_x -left_side
        
        
        # Creating the cutout or snapshot of the twins in TI
        snapshot=TI_mosaic[bottom:top, left_side:right_side]
        # Creating the cutout or snapshot of the twins in PI
        PI_snapshot = PI_mosaic[bottom:top, left_side:right_side]
        
        # Creating an array of pixel values to use in the snapshot plot
        ticks_cutout_x= [n for n in range(0, len(snapshot[0]),1)]
        ticks_cutout_y= [n for n in range(0, len(snapshot[:,0]), 1)]
        
        
        
        # Getting the galactic coordinates
        all_xlabels, all_ylabels= fc.pixel_to_galactic_coordinates_axis(mosaic)
        
        
        # Creating an array with the pixel values of the snapshot
        reduced_y_ticks= np.arange(bottom, top, 1)
        reduced_x_ticks = np.arange(left_side,right_side, 1)
        
       
        
        # Getting the galactic coordinates of the snapshot
        cutout_xlabels= np.round(all_xlabels[reduced_x_ticks],2)
        cutout_ylabels = np.round(all_ylabels[reduced_y_ticks], 2)
        
        
        if plot_AGNs:
            AGN_cutout_xlabels,AGN_cutout_ylabels  = np.round(all_xlabels[reduced_x_ticks],3),\
                                                        np.round(all_ylabels[reduced_y_ticks],3)
            
            lmin,lmax = AGN_cutout_xlabels[-1],AGN_cutout_xlabels[0]
            bmin, bmax = AGN_cutout_ylabels[0], AGN_cutout_ylabels[-1]
            
           

        # Getting the galactic coordinates of the PI intensity twin coordinates.
        t1PI_y_GalCoord, t1PI_x_GalCoord, t1PI_r_GalCoord = all_ylabels[int(t1PI_y)],\
            all_xlabels[int(t1PI_x)], t1PI_r*17.95688 #1pixel = 17.95688"
        t2PI_y_GalCoord, t2PI_x_GalCoord, t2PI_r_GalCoord = all_ylabels[int(t2PI_y)],\
            all_xlabels[int(t2PI_x)], t2PI_r*17.95688 #1pixel = 17.95688"
            
        # Getting the center in galactic coordinates    
        center_lat, center_long = all_ylabels[int(c_y)],  all_xlabels[int(c_x)]
        center_gal = [all_ylabels[int(c_y)],  all_xlabels[int(c_x)]]

        
        # Getting the distance between lobes in arcminutes
        distance_arcmin = (17.95688/60) *distance[n]
        
        
        if plot_snapshots:
            
            

            #Creating the plot
            
            fig= plt.figure(figsize=(12,8), constrained_layout=False)
            fig.suptitle("Pair " + str(n)+" in mosaic "+ str(mosaic.upper()[1:]), fontsize=fontsize*2)
            
            # Specifying the size of the whole plot
            spec = fig.add_gridspec(nrows=7, ncols=12)
            
            #Specifying the size of the first plot
            ax0= fig.add_subplot(spec[:-5, :-8])
            # Setting the parameters of the first plot
            ax0.set_title("Stokes I", fontsize=fontsize)
            ax0.set_xticks(ticks_cutout_x, labels=cutout_xlabels)
            ax0.set_yticks(ticks_cutout_y, labels=cutout_ylabels)
            ax0.xaxis.set_major_locator(plt.MaxNLocator("auto"))
            ax0.yaxis.set_major_locator(plt.MaxNLocator("auto"))
      
            ax0.set_xlabel(r"Longitude $(^\circ)$")
            ax0.set_ylabel (r"Latitude $(^\circ)$")
            
            PLT= ax0.imshow(snapshot,  vmin = pf.TI_VMIN, vmax = pf.TI_VMAX,cmap="gist_heat",origin='lower')
            cbar=fig.colorbar(PLT)
            ticksforcolorbar = np.linspace(pf.TI_VMIN,pf.TI_VMAX, 6)
            cbar.set_ticks(ticksforcolorbar.tolist())
            
            
            # Specify the size of the 2D Polarized intensity plot
            ax1 = fig.add_subplot(spec[:-5, -7:-1])
            
            ax1.set_title("Polarised Intensity", fontsize=fontsize)
            ax1.set_xticks(ticks_cutout_x, labels=cutout_xlabels)
            ax1.set_yticks(ticks_cutout_y, labels=cutout_ylabels)
            ax1.xaxis.set_major_locator(plt.MaxNLocator("auto"))
            ax1.yaxis.set_major_locator(plt.MaxNLocator("auto"))
            # ax1.set_xticklabels(cutout_xlabels)
      
            ax1.set_xlabel(r"Longitude $(^\circ)$")
            ax1.set_ylabel (r"Latitude $(^\circ)$")
            
            PLT= ax1.imshow(PI_snapshot,  vmin = pf.PI_VMIN, vmax = pf.PI_VMAX,cmap="gist_heat",origin='lower')
            cbar=fig.colorbar(PLT)
            ticksforcolorbar = np.linspace(pf.PI_VMIN,pf.PI_VMAX, 6)
            cbar.set_ticks(ticksforcolorbar.tolist())
            
            QSO_source, AGN_source,DRAGN_source, G_source, LrG_source  = False, False, False, False,False
            
            
            if plot_AGNs:
                QSOsarray, dfQSOs = fc.get_AGNs_CatNorth(lmin, lmax, bmin, bmax, mosaic)
                QSOs= np.round(QSOsarray,3)
                
                
                for Q in QSOs:
                        
                     l, b= Q
                     
                     x,y  = np.nanargmin(np.abs(AGN_cutout_xlabels-l)),np.nanargmin(np.abs(AGN_cutout_ylabels-b))
                     
                     # cir = 0
                     cir1 = plt.Circle((x,y), radius = 3,  color= "gold", linewidth = 2, fill=1, alpha=0.6)
                     cir2 = plt.Circle((x,y), radius = 3,  color= "gold", linewidth = 2, fill=2, alpha=0.6)
                     ax0.add_patch(cir1)
                     ax1.add_patch(cir2)
                     QSO_source=True
                
                
                
                
                DRAGNsarray, dfDRAGNs = fc.get_AGNs_VLA_DRAGNs(lmin, lmax, bmin, bmax)
                DRAGNs = np.round(DRAGNsarray, 3)
               
               
                for D in DRAGNs:
                    
                     l, b= D
                     
                     x,y  = np.nanargmin(np.abs(AGN_cutout_xlabels-l)),np.nanargmin(np.abs(AGN_cutout_ylabels-b))
                     # cir = 0
                     cir1 = plt.Circle((x,y), radius = 3,  color= "hotpink", linewidth = 2, fill=1, alpha=0.6)
                     cir2 = plt.Circle((x,y), radius = 3,  color= "hotpink", linewidth = 2, fill=1, alpha=0.6)
                     ax0.add_patch(cir1)
                     ax1.add_patch(cir2)
                     # print(len(GPQs) , " Quasar from GP")
                     DRAGN_source=True
                     
                
                # Getting the AGNs in the mosaics
                AGNsarray, dfAGNs = fc.get_AGNs_simbad(lmin, lmax, bmin, bmax)
                AGNs = np.round(AGNsarray,3)
                
             
                
                

                
                    
                for AGN in AGNs:
                    l, b= AGN
                    
                    x,y  = np.nanargmin(np.abs(AGN_cutout_xlabels-l)),np.nanargmin(np.abs(AGN_cutout_ylabels-b))
                    # print("l2146 x,y: ", x,y)
                    cir1 = plt.Circle((x,y), radius = 2,  color= "darkgrey", linewidth = 2, fill=1, alpha=0.6)
                    cir2 = plt.Circle((x,y), radius = 2,  color= "darkgrey", linewidth = 2, fill=1, alpha=0.6)
                    ax0.add_patch(cir1)
                    ax1.add_patch(cir2)
                    AGN_source=True
                print(lmin, lmax, bmin, bmax)
                LrGarray, dfLrGs = fc.get_AGNs_VLA_large_rG(lmin, lmax, bmin, bmax)
                print(dfLrGs)
                LrGs = np.round(LrGarray, 3)
                
                for LrG in LrGs:
                    l, b= LrG
                    
                    x,y  = np.nanargmin(np.abs(AGN_cutout_xlabels-l)),np.nanargmin(np.abs(AGN_cutout_ylabels-b))
                    # print("l2146 x,y: ", x,y)
                    cir1 = plt.Circle((x,y), radius = 2,  color= "mediumblue", linewidth = 2, fill=1, alpha=0.6)
                    cir2 = plt.Circle((x,y), radius = 2,  color= "mediumblue", linewidth = 2, fill=1, alpha=0.6)
                    ax0.add_patch(cir1)
                    ax1.add_patch(cir2)
                    LrG_source=True
                
               
                
                
                
                        
                
                
                if QSO_source+AGN_source+DRAGN_source+G_source +LrG_source==0:#means no sources associated from catalogue
                    associated_AGN = False
                    offset_AGN=name_AGN=name_AGN= AGN_long=AGN_lat= survey_AGN=None
                
                else: 
                   all_databases= pd.concat([dfQSOs, dfDRAGNs, dfAGNs, dfLrGs])
         


                    
                
                
        # Identifying all the sources in the cutout
        TI_cutout_sources = fc.Identify_Point_Sources(snapshot, vmin=pf.TI_VMIN,
                                                      vmax=pf.TI_VMAX, threshold=3*min_PI_peak/4, plot=False, )
        
        
        # Getting just the radii of the sources, 
        TI_radii = np.copy(TI_cutout_sources[:,2])
        
       
        # Creating a list to store the sources that radii are within the max and min radii
        reduce_TI_cutout_sources=[]
        for N, r in enumerate(TI_radii):
            if r<pf.max_TI_radius and r>pf.min_TI_radius:
                # reduce_TI_cutout_sources += [TI_cutout_sources.tolist()[N]]
                y, x, HWHM = TI_cutout_sources[N]
                
               
                # Making sure the sources aren't on the edge of the image
                if (y>=2 and y<=snapshot_full_length-2 and x>=2 and 
                    x<=snapshot_full_length-2):  
                    peak = snapshot[int(y),int(x)]
                    source = [y, x, HWHM, peak]
                    reduce_TI_cutout_sources += [source]
                    
               
                
                
                
                
        
        
       
        
        # Classifying Via the algorithm all the sources in the cutout
        binary_classification, TI_twins, all_twin_sources, solo_sources, central_sources\
            = TI_twin_detector_and_binary_pair_classifiers(np.array(reduce_TI_cutout_sources),\
                                                           c, left_side=left_side, bottom=bottom, twinPI=twin_list[n],)
                
        
        
     
        

        
#### Adjusting the coordinates so that they will be the coordinates for the mosaic not the cut out
        
        # adjusting the coordinates of the detected sources only if there were sources detected. 
        if len(reduce_TI_cutout_sources) != 0:
        
            # Doing the same for the detected sources 
            detected_sources_array = np.array(TI_cutout_sources)
            
            detected_sources_array[:,0] += bottom
            detected_sources_array[:,1] +=  left_side
           
            detected_sources_list = detected_sources_array.tolist()


        # converting the TI_twins list to an array to make adjusting the coordinates easier and faster
        TI_Twins_array = np.array(TI_twins)
        # Only changing the coordinates if the twins list is not empty
        if TI_Twins_array.shape != (1,1,0) and len(TI_Twins_array)!=0:
            
            # Adding the bottom and left edge pixel coordinates to convert from the 
            #   snapshot coordinates to the full mosaic pixel coordinates. 
            TI_Twins_array[0,:,0] += bottom
            TI_Twins_array[0,:,1] +=  left_side
            
            # Converting back into a list
            TI_twins = TI_Twins_array.tolist()
            
            
            
            
            # Doing the same but for the individual sources 
            all_twin_s_array = np.array(all_twin_sources)
            
            all_twin_s_array[:,0] += bottom
            all_twin_s_array[:,1] += left_side
            
            all_twin_sources = all_twin_s_array.tolist()
        
            
        # If there is a twin pair, there will not be any central sources. 
        elif len(central_sources) !=0:
            central_sources_array = np.array(central_sources)
            central_sources_array[:,0] += bottom
            central_sources_array[:,1] += left_side
            central_sources = central_sources_array.tolist()
        
        # changing the coordinates for the solo sources detected.    
        if len(solo_sources) != 0:
            solo_s_array = np.array(solo_sources)
            solo_s_array[:,0] += bottom
            solo_s_array[:,1] += left_side
            solo_sources = solo_s_array.tolist()
        
        
        
   #### This section of code converts the Total Intensity pixel coordinates to  Galactic coordinates.    
        # Creating all the empty lists for the conversion 
        TI_twins_gal_coor, all_TI_twins_sources_gal_coor, central_sources_gal_coor,\
        solo_sources_gal_coor =[], [],[], []
        # looping through all the twin pairs in total intensity
        for t in TI_twins:
            t1, t2 = t
            # Creating individual variables for the x,y,radial, and peak values of each twin
            TI_y1, TI_x1, TI_r1, TI_p1 = t1
            TI_y2, TI_x2, TI_r2, TI_p2 = t2
            
            # getting the galactic coordinates 
            TI_y_gal_coord1, TI_x_gal_coord1, TI_r_arcsec1 = \
                all_ylabels[int(TI_y1)], all_xlabels[int(TI_x1)], TI_r1*17.95688 #1pixel = 17.95688"
            TI_y_gal_coord2, TI_x_gal_coord2, TI_r_arcsec2 = \
                all_ylabels[int(TI_y2)], all_xlabels[int(TI_x2)], TI_r2*17.95688 #1pixel = 17.95688"
            
            # Adding the galactic coordinates to the appropriate list
            TI_twins_gal_coor.append([[TI_y_gal_coord1, TI_x_gal_coord1, TI_r_arcsec1, TI_p1],
                                      [TI_y_gal_coord2, TI_x_gal_coord2, TI_r_arcsec2, TI_p2]])
            # Adding the galactic coordinates to the appropriate list
            all_TI_twins_sources_gal_coor.append([TI_y_gal_coord1, TI_x_gal_coord1, TI_r_arcsec1, TI_p1])
            all_TI_twins_sources_gal_coor.append([TI_y_gal_coord2, TI_x_gal_coord2, TI_r_arcsec2, TI_p2])
        
        # Note there should not be any central sources if there is a twin pair, and vise versa.
        for s in central_sources:
            # Setting the coordinates values for the central source (one singular source 
            #   associated with both PI sources) 
            TI_y1, TI_x1, TI_r1, TI_p1 = s
            TI_y2, TI_x2, TI_r2, TI_p2 = None, None, None, None
            
            # getting the galactic coordinates 
            TI_y_gal_coord1, TI_x_gal_coord1, TI_r_arcsec1 = \
                all_ylabels[int(TI_y1)], all_xlabels[int(TI_x1)], TI_r1*17.95688 #1pixel = 17.95688"
            TI_y_gal_coord2, TI_x_gal_coord2, TI_r_arcsec2, TI_p2 = None, None, None, None
            
            # Adding the galactic coordinates to the appropriate list
            central_sources_gal_coor.append([TI_y_gal_coord1, TI_x_gal_coord1, TI_r_arcsec1, TI_p1])
            central_sources_gal_coor.append([TI_y_gal_coord2, TI_x_gal_coord2, TI_r_arcsec2, TI_p2])
        
        # Putting up an error up if there is both single source and a twin source detected. 
        if len(TI_twins) != 0 and len(central_sources) != 0:
            print("Huston we have a problem: twins and central sources")
        
        # getting the coordinates and galactic coordinates of the solo sources 
        #   if there are no twin or central sources the following if is there is 
            #   the following if is there is only 1 solo source
        elif len(TI_twins) ==0  and len(central_sources) ==0 and len(solo_sources) ==1:
            
            # Setting the coordinates values for the central source (one singular source 
            #   associated with both PI sources) 
            TI_y1, TI_x1, TI_r1, TI_p1 = solo_sources[0]
            TI_y2, TI_x2, TI_r2, TI_p2 = None, None, None, None
            
            # getting the galactic coordinates 
            TI_y_gal_coord1, TI_x_gal_coord1, TI_r_arcsec1 = \
                all_ylabels[int(TI_y1)], all_xlabels[int(TI_x1)], TI_r1*17.95688 #1pixel = 17.95688"
            TI_y_gal_coord2, TI_x_gal_coord2, TI_r_arcsec2, TI_p2 = None, None, None, None
                # all_ylabels[int(TI_y2)], all_xlabels[int(TI_x2)], TI_r2*17.95688 #1pixel = 17.95688"
            
            # Adding the galactic coordinates to the appropriate list
            solo_sources_gal_coor.append([TI_y_gal_coord1, TI_x_gal_coord1, TI_r_arcsec1, TI_p1])
            
            #   the following if is there is more than 1 solo source
        elif len(TI_twins) ==0 and len(solo_sources) >1:
            
            # Setting the coordinates values for the central source (one singular source 
            #   associated with both PI sources)
            TI_y1, TI_x1, TI_r1, TI_p1 = solo_sources[0]
            TI_y2, TI_x2, TI_r2, TI_p2 = solo_sources[1]
            
            # getting the galactic coordinates 
            TI_y_gal_coord1, TI_x_gal_coord1, TI_r_arcsec1 = \
                all_ylabels[int(TI_y1)], all_xlabels[int(TI_x1)], TI_r1*17.95688 #1pixel = 17.95688"
            TI_y_gal_coord2, TI_x_gal_coord2, TI_r_arcsec2 = \
                all_ylabels[int(TI_y2)], all_xlabels[int(TI_x2)], TI_r2*17.95688 #1pixel = 17.95688"
            
            # Adding the galactic coordinates to the appropriate list
            solo_sources_gal_coor.append([TI_y_gal_coord1, TI_x_gal_coord1, TI_r_arcsec1, TI_p1])
            solo_sources_gal_coor.append([TI_y_gal_coord2, TI_x_gal_coord2, TI_r_arcsec2, TI_p2])
     
            # Looping through the remaining solo sources to add to the galactic coordinates list
            for i in range(2,len(solo_sources)):
                TI_y,TI_x,TI_r,TI_p = solo_sources[i]
                TI_y_gal, TI_x_gal, TI_r_gal = all_ylabels[int(TI_y)], all_xlabels[int(TI_x)], TI_r*17.95688 #1pixel = 17.95688"
                
                solo_sources_gal_coor.append([TI_y_gal, TI_x_gal, TI_r_gal, TI_p])
                
        
        elif len(TI_twins) ==0 and len(solo_sources)==0 and len(central_sources)==0:
            
            TI_y1, TI_x1, TI_r1, TI_p1 = None, None, None, None
            TI_y2, TI_x2, TI_r2, TI_p2 = None, None, None, None
            
            # getting the galactic coordinates 
            TI_y_gal_coord1, TI_x_gal_coord1, TI_r_arcsec1 = None, None, None
            TI_y_gal_coord2, TI_x_gal_coord2, TI_r_arcsec2 = None, None, None
        
        
            
            
            
        StoN_twins = [[t1_StoN, t2_StoN]]     
            
        
        if n ==0: # If this is the first source examined and the lists need to be initilazed 
           # creating a classification list 
           if binary_classification==None: binary_classification="None"
           mosaic_pair_classifications = [binary_classification]
           
           if return_singular_list ==False:
               # Adding the pair detections to the source lists
                mosaic_TI_twins = TI_twins.copy()
                mosaic_solo_sources = solo_sources.copy()
                mosaic_central_sources = central_sources.copy()
                detected_sources_list =detected_sources_array.tolist()
                all_mosaic_twin_sources = all_twin_sources.copy()
                mosaic_TI_twins_gal_coord = TI_twins_gal_coor.copy()
                mosaic_TI_twins_sources_gal_coord = all_TI_twins_sources_gal_coor.copy()
                mosaic_solo_sources_gal_coord = solo_sources_gal_coor.copy()
                mosaic_central_sources_gal_coor = central_sources_gal_coor.copy()
                mosaic_Signal_to_Noise = StoN_twins.copy()
                mosaic_centers_gal = center_gal.copy()
                mosaic_distances_pix = [distance[n]]
                mosaic_distances_arcmin = [distance_arcmin]
                mosaic_associated_AGNs = [associated_AGN]
                mosaic_sources_offset = [offset_AGN]
                mosaic_AGN_names = [name_AGN]
                mosaic_AGN_long = [AGN_long]
                mosaic_AGN_lat=[AGN_lat]
                mosaic_AGN_surveys = [survey_AGN]
            
        else: 
           if binary_classification==None: binary_classification="None"
           mosaic_pair_classifications.append(binary_classification)
           
           if return_singular_list ==False:
               # Adding the pair detections to the source lists
                mosaic_TI_twins += TI_twins.copy()
                mosaic_solo_sources += solo_sources.copy()
                mosaic_central_sources += central_sources.copy()
                detected_sources_list += detected_sources_array.tolist()
                all_mosaic_twin_sources += all_twin_sources.copy()
                mosaic_TI_twins_gal_coord += TI_twins_gal_coor.copy()
                mosaic_TI_twins_sources_gal_coord += all_TI_twins_sources_gal_coor.copy()
                mosaic_solo_sources_gal_coord += solo_sources_gal_coor.copy()
                mosaic_central_sources_gal_coor += central_sources_gal_coor.copy()
                mosaic_Signal_to_Noise += StoN_twins.copy()
                mosaic_centers_gal += center_gal.copy()
                mosaic_distances_pix += [distance[n]]
                mosaic_distances_arcmin += [distance_arcmin]
                mosaic_associated_AGNs += [associated_AGN]
                mosaic_sources_offset += [offset_AGN]
                mosaic_AGN_names += [name_AGN]
                mosaic_AGN_long += [AGN_long]
                mosaic_AGN_lat+=[AGN_lat]
                mosaic_AGN_surveys += [survey_AGN]
        
                
    
      
        
        if plot_snapshots:
            
            # Adding all the sources found in TI to the 2D plot of Stokes I 
            for s in np.array(reduce_TI_cutout_sources):
                sy,sx,sr,sp = s 
                
                circle= plt.Circle((sx,sy), pf.radius_scale*sr, color="cyan", fill=False, linewidth=2)
                
                ax0.add_patch(circle)
            
            
            # Getting the initial (unrounded) coordinates
            (t1y_i,t1x_i,t1r, t1p), (t2y_i, t2x_i, t2r, t2p) = twin1PI, twin2PI
  
            # Rounding the coordinates so they can be aside a pixel value. 
            t1y, t2y, t1x,t2x = round(t1y_i) - bottom, round(t2y_i) - bottom,\
                round(t1x_i) -left_side, round(t2x_i) -left_side
            
            # Creating the Circles for PI 2D plot
            circle1= plt.Circle((t1x,t1y), pf.radius_scale*t1r, color="lime", fill=False, linewidth=2)
            circle2= plt.Circle((t2x,t2y), pf.radius_scale*t2r, color="lime", fill=False, linewidth=2)
         
            # Creating the Circles for Stokes I 2D plot
            circle3= plt.Circle((t1x,t1y), pf.radius_scale*t1r, color="lime", fill=False, linewidth=2, linestyle=":")
            circle4= plt.Circle((t2x,t2y), pf.radius_scale*t2r, color="lime", fill=False, linewidth=2, linestyle=":")
         
            
            # Adding the circles to the plots
            ax1.add_patch(circle1)
            ax1.add_patch(circle2)
            ax0.add_patch(circle3)
            ax0.add_patch(circle4)
            
            
    
            central_classes= [17, 9, 18,146,20]
           
            if binary_classification in central_classes:
                offset_center = plt.Circle((x_center_TI,y_center_TI), pf.max_offset, color="silver", fill=False, linewidth=2, linestyle="--")
                ax0.add_patch(offset_center)
            else:
                
                # Creating the circles for the offset/correlation region in 
                Offset_region1 = plt.Circle((t1x,t1y), pf.max_offset, color="silver", fill=False, linewidth=2, linestyle="--")
                Offset_region2 = plt.Circle((t2x,t2y), pf.max_offset, color="silver", fill=False, linewidth=2, linestyle="--")
                
                ax0.add_patch(Offset_region1)
                ax0.add_patch(Offset_region2)
            
            
            
            # Specify the placement and size of the three dimensional plots
            ax2 = fig.add_subplot(spec[-4:, :-6], projection='3d')
            ax3 = fig.add_subplot(spec[-4:,-6:], projection='3d')

           
            
            
            # Creating the 3-D plots
            y = range( snapshot.shape[0] )
            x = range( snapshot.shape[1] ) 
            X, Y = np.meshgrid(x, y)
            
            
            plot3d = ax2.plot_surface( X, Y, snapshot, cmap=plt.colormaps["gist_heat"],
                                     vmax=pf.TI_VMAX, vmin=pf.TI_VMIN)
            plot3d_PI = ax3.plot_surface( X, Y, PI_snapshot, cmap=plt.colormaps["gist_heat"],
                                     vmax=pf.PI_VMAX, vmin=pf.PI_VMIN)
            
            # Adding the circles (or cylinders) onto the 3-D floss, if selected
            if plot_detection_circles:
                
                # Adding the cylinders from the sources detected in Stokes I 
                for s in np.array(reduce_TI_cutout_sources):
                    
                    sy,sx,sr, sp = s 
                    # I can't find the OG code I used to create the cylinders, but was something similar to this. 
                    #   https://scipython.com/book/chapter-7-matplotlib/examples/a-torus/
                    R = pf.radius_scale*sr
                    
                    angle = np.linspace(0, 2 * np.pi, 100)
                    theta, phi = np.meshgrid(angle, angle)
                    r = .25
                    X = (R + r * np.cos(phi)) * np.cos(theta) +sx
                    Y = (R + r * np.cos(phi)) * np.sin(theta)+sy
                   
                    Z = snapshot.max() * np.sin(phi)*0.7 +snapshot.max()/2 -0.01
                    
                    
                    ax2.plot_surface(X, Y, Z, color = 'cyan', alpha=1)#0.5)
                    
                    
                # Adding the cylinders from the sources detected in PI
               
                (t1y_i,t1x_i,t1r, t1p), (t2y_i, t2x_i, t2r, t2p) = twin1PI, twin2PI
                
                t1y, t2y, t1x,t2x = round(t1y_i) - bottom, round(t2y_i) - bottom,\
                    round(t1x_i) -left_side, round(t2x_i) -left_side
                
                R1 = pf.radius_scale*t1r
                R2 = pf.radius_scale*t2r
                
                angle = np.linspace(0, 2 * np.pi, 32)
                theta, phi = np.meshgrid(angle, angle)
                r = .25
                X1 = (R1 + r * np.cos(phi)) * np.cos(theta) +t1x
                Y1 = (R1 + r * np.cos(phi)) * np.sin(theta)+t1y
                Z1 = PI_snapshot.max() * np.sin(phi)*0.55 +PI_snapshot.max()/2 -PI_snapshot.max()/100
                
                X2 = (R2 + r * np.cos(phi)) * np.cos(theta) +t2x
                Y2 = (R2 + r * np.cos(phi)) * np.sin(theta)+t2y
                Z2 = PI_snapshot.max() * np.sin(phi)*0.55 +PI_snapshot.max()/2 -PI_snapshot.max()/100
               
                
                ax3.plot_surface(X1, Y1, Z1, color = 'lime', alpha=1)
                ax3.plot_surface(X2, Y2, Z2, color = 'lime', alpha=1)
                    
                    

                    
            # Making the plots look good
            ax2.set_title("Stokes I", fontsize=fontsize)
            ax2.set_xticks(ticks_cutout_x, labels=cutout_xlabels)
            ax2.set_yticks(ticks_cutout_y, labels=cutout_ylabels)
            ax2.xaxis.set_major_locator(plt.MaxNLocator("auto"))
            ax2.yaxis.set_major_locator(plt.MaxNLocator("auto"))
            
            # Changing the pane color to be darker. 
            ax2.xaxis.set_pane_color((0.5,0.5,0.5,1))
            ax2.yaxis.set_pane_color((0.5,0.5,0.5,1))
            ax2.zaxis.set_pane_color((0.5,0.5,0.5,1))
      
            ax2.set_xlabel(r"Longitude $(^\circ)$")
            ax2.set_ylabel (r"Latitude $(^\circ)$")
            ax2.set_zlabel("Total Intensity")
            
            ax2.set_zlim(0,snapshot.max()+0.01)
            cbar=fig.colorbar(plot3d, shrink=0.4, pad=0.15)#, pad=0.3)
            ticksforcolorbar = np.linspace(pf.TI_VMIN,pf.TI_VMAX, 6)
            cbar.set_ticks(ticksforcolorbar.tolist())
            
            
            ax3.set_title("Polarised Intensity", fontsize=fontsize)
            ax3.set_xticks(ticks_cutout_x, labels=cutout_xlabels)
            ax3.set_yticks(ticks_cutout_y, labels=cutout_ylabels)
            ax3.xaxis.set_major_locator(plt.MaxNLocator("auto"))
            ax3.yaxis.set_major_locator(plt.MaxNLocator("auto"))
            
            # Changing the pane color to be darker. 
            ax3.xaxis.set_pane_color((0.5,0.5,0.5,1))
            ax3.yaxis.set_pane_color((0.5,0.5,0.5,1))
            ax3.zaxis.set_pane_color((0.5,0.5,0.5,1))
      
            ax3.set_xlabel(r"Longitude $(^\circ)$")
            ax3.set_ylabel (r"Latitude $(^\circ)$")
            ax3.set_zlabel("Polarised Intensity")
            
            ax3.set_zlim(0,PI_snapshot.max()+0.001)
            cbar=fig.colorbar(plot3d_PI, shrink=0.4, pad=0.15)
            ticksforcolorbar = np.linspace(pf.PI_VMIN,pf.PI_VMAX, 6)
            cbar.set_ticks(ticksforcolorbar.tolist())
            
            plt.savefig(out_dir +"missed_rG_"+"Pair_" +str(n)+".pdf", format="pdf")
            plt.savefig(out_dir +"missed_rG_"+"Pair_" +str(n)+".svg", format="svg")
            
            # The following function forces the plots to open in line in the external
            #   graphic producer, rather than opening at the end of the script. 
            plt.pause(initial_graph_time)
        
            
     
        print("The radii in PI of the sources: ", t1PI_r, " and ", t2PI_r)
       
        # Code for adding time to look and interact with the plots. 
        print("Pair classification for snapshot ", n, " is: ", binary_classification)
        
        if plot_snapshots:
            # Getting the user's input for the true classification of the twin
            true_classification = input('''Please enter the classification of the twin: (Enter "?" for more time): ''')
            if true_classification == "":
                true_classification = binary_classification

            
            elif true_classification != "?":
                try:
                    # Setting the correct classification based on the input since it is automatically a string. 
                    if true_classification == str(None):
                        true_classification = "None"
                    elif true_classification == str(False):
                        true_classification = False
                    else:
                        true_classification= int(true_classification)
                        
                # Raising an error if an invalid entry is given (not None, False or an integer)
                except:
                    print("ERROR: Invalid entry, please enter a valid class:")
                    true_classification= input("Please enter the classification of the twin:")
                    if true_classification == str(None):
                        true_classification = "None"
                    elif true_classification == str(False):
                        true_classification = False
                    else:
                        true_classification= int(true_classification)
                    
                
                
            else: # Adding more time to look at the graphs if specified. 
                
            
                more_time = input("Enter the number of seconds you wish to able to manipulate the graphs: ")
                
                try:
                    float(more_time)
                except: # Raising an error if an invalid number of seconds is given
                    print("Invalid entry")
                    more_time = float(input("Enter the number of seconds you wish to able to manipulate the graphs: "))
                    
                more_time = float(more_time)
                
                
                if more_time>40: # Checking to make sure you want the larger amount 
                #   of time enter (in case you enter the class, like 546, of the twin instead.)
                    check = input("Are you sure you want "+ str(more_time)+" seconds to look at the graph (enter y or n):")
                    if check =="n" or check == "N" or check == "no" or check=="0":
                        more_time = float(input("Please enter the amount of time you want: "))
                        
                        
                while more_time !=0:
                    
                    plt.pause(more_time)
                    # more_time = input("Enter the number of seconds you wish to able to manipulate the graphs: ")
                    
                    try:
                        more_time = float(input("Enter the number of seconds you wish to able to manipulate the graphs: "))
                        
                    except:
                        print("Invalid entry")
                        more_time = float(input("Enter the number of seconds you wish to able to manipulate the graphs: "))
                        
                    if more_time>40:
                        check = input("Are you sure you want "+ str(more_time)+" seconds to look at the graph (enter y or n):")
                        if check =="n" or check == "N" or check == "no"or check=="0":
                            more_time = float(input("Please enter the amount of time you want: "))
                try:
                    Class = input("Please enter the classification of the twin:")

                    if Class == str(None):
                        true_classification = None
                    elif Class == str(False):
                        true_classification = False
                    else:
                        true_classification= int(Class)
                except:
                    print("ERROR: Invalid entry, please enter an interger number")
                    Class = input("Please enter the classification of the twin:")
                    if Class == str(None):
                        true_classification = None
                    elif Class == str(False):
                        true_classification = False
                    else:
                        true_classification= int(Class)
        
            
        # Checking if the entered classification is a valid classification. 
        dict_words, dict_keys = binary_classification_dictionaries()
        
        
        # Getting the different classes and what they indicate 
        if true_classification not in list(dict_words.keys()):
            print("ERROR: Invalid classification number")
            Class = input("Please enter the classification of the twin:")
            if Class == str(None):
                true_classification = None
            elif Class == str(False):
                true_classification = False
            else:
                true_classification= int(Class)
            
        if n==0:
            mosaic_true_classifications = [true_classification]
        else:
            mosaic_true_classifications.append(true_classification)
        
        
        if binary_classification == 546 or binary_classification == 548 or binary_classification == 580:
            twin_detected_auto = True
        else:
            twin_detected_auto = False
            
        if true_classification == 546 or true_classification == 548 or true_classification == 580:
            true_twin_detected = True
        else:
            true_twin_detected = False
        
        
        
        
        
        twin_data  = [mosaic, twin_detected_auto, true_twin_detected, binary_classification, \
                      true_classification, t1PI_p, t2PI_p, TI_p1, TI_p2, t1PI_y, t1PI_x, t1PI_r,\
                        t2PI_y, t2PI_x, t2PI_r, t1PI_y_GalCoord, t1PI_x_GalCoord, t1PI_r_GalCoord,\
                        t2PI_y_GalCoord, t2PI_x_GalCoord, t2PI_r_GalCoord, TI_y1, TI_x1, TI_r1,\
                        TI_y2, TI_x2, TI_r2, TI_y_gal_coord1, TI_x_gal_coord1, TI_r_arcsec1, \
                        TI_y_gal_coord2, TI_x_gal_coord2, TI_r_arcsec2, t1_StoN, t2_StoN, 
                        c_y, c_x, center_lat, center_long, distance[n], distance_arcmin,
                        associated_AGN, offset_AGN, name_AGN, AGN_long, AGN_lat, survey_AGN]
        
        twin_dataset.append(twin_data)
    print("mosaic_pair_classifications: ", mosaic_pair_classifications)
    
    print("mosaic_true_classificaations: ", mosaic_true_classifications)
    
    
    if return_singular_list:
        return twin_dataset
    else:
        return (mosaic_TI_twins, mosaic_solo_sources, mosaic_pair_classifications, 
                    mosaic_true_classifications, detected_sources_list, all_mosaic_twin_sources,
                    mosaic_TI_twins_gal_coord, mosaic_TI_twins_sources_gal_coord, 
                    mosaic_solo_sources_gal_coord, mosaic_Signal_to_Noise, mosaic_centers_gal,
                    mosaic_distances_pix, mosaic_distances_arcmin, mosaic_associated_AGNs,
                    mosaic_sources_offset,mosaic_AGN_names, mosaic_AGN_long,mosaic_AGN_lat, mosaic_AGN_surveys  )





def write_dat_file(mosaic, mosaic_dataset,):
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
    mosaic_array = np.array(mosaic_dataset)
    
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
    
    
    # Determining if the detected sources were twins 
    Twin_Detected = mosaic_array[:,2]#.tolist() # true twin detected column
    
    
    # Getting the twin 1 values for each pair in the mosaic
    t1_gal_long = np.array(mosaic_array[:, 16] ,dtype="float") # Galactic longitude of twin 1
    t1_gal_lat = np.array(mosaic_array[:, 15],dtype="float") # Galactic latitude of twin 1 (T1)
    t1_xpix =  np.array(mosaic_array[:, 10],dtype="float") # the x pixel coordinate of T1 in the mosaic
    t1_ypix = np.array(mosaic_array[:, 9],dtype="float") # the # the x pixel coordinate of T1 in the mosaic pixel coordinate of T1 in the mosaic
    t1_PI = np.array(mosaic_array[:, 5],dtype="float")*1000 # getting polarised intensity (the 1000 converts from Jy\beam to mJy\beam)
    t1_TI = np.array(mosaic_array[:, 7],dtype="float")*1000 # getting total intensity (the 1000 converts from Jy\beam to mJy\beam)
    t1_SN = np.array(mosaic_array[:, 33],dtype="float") # Getting the signal to noise of Twin 1
    
    # Getting the twin 2 values for each pair in the mosaic, same values as T1
    t2_gal_long = np.array(mosaic_array[:, 19],dtype="float")
    t2_gal_lat = np.array(mosaic_array[:, 18],dtype="float")
    t2_xpix =  np.array(mosaic_array[:, 13],dtype="float")
    t2_ypix = np.array(mosaic_array[:, 12],dtype="float")
    t2_PI = np.array(mosaic_array[:, 6],dtype="float")*1000# Convert from Jy\beam to mJy\beam
    t2_TI = np.array(mosaic_array[:, 8],dtype="float")*1000# Convert from Jy\beam to mJy\beam
    t2_SN = np.array(mosaic_array[:, 34],dtype="float")
    
    
    
    
    
# =============================================================================
#     # Writing the file
# =============================================================================
    with open(f'{out_dir}/{mosaic.upper()}_twins.dat', "w") as write_twins:
        write_twins.write(f'Polarized twin source candidate list for field {mosaic.upper()}')
        write_twins.write('(\nGenerated using Ciara Chisholms twin source detection algorithm).')
        write_twins.write('\n ')
        write_twins.write('\n   l       b     xpix  ypix       PI       SI      S/N')
        write_twins.write('\n-- degrees --                     --mJy/beam--')
        write_twins.write('\n ')
        
        # Going through all the sources in the data
        for p,t in enumerate(Twin_Detected):
            # Only writing a line if a twin was detected 
            if t:
                
                # Getting the PI peak of the sources 
                t1_PIpeak,t2_PIpeak = round(t1_PI[p], 2),round(t2_PI[p], 2)
                
                # Writing the largest PI peak first 
                if t1_PIpeak >= t2_PIpeak:
                    # Writing twin 1's info first (since it has the higher PI peak). 
                    #   {variable:num_of_characters.decimals_places f } the f is for float, if the there are more characters than the number assigned, it will add spaces in front of the number. 
                    write_twins.write(f'\n{t1_gal_long[p]:7.3f}'
                                      f'{t1_gal_lat[p]:7.3f}'
                                      f'{t1_xpix[p]:7.0f}'
                                      f'{t1_ypix[p]:6.0f}'
                                      f'{t1_PI[p]:10.2f}'
                                      f'{t1_TI[p]:11.2f}'
                                      f'{t1_SN[p]:7.2f}')
                    write_twins.write(f'\n{t2_gal_long[p]:7.3f}'
                                      f'{t2_gal_lat[p]:7.3f}'
                                      f'{t2_xpix[p]:7.0f}'
                                      f'{t2_ypix[p]:6.0f}'
                                      f'{t2_PI[p]:10.2f}'
                                      f'{t2_TI[p]:11.2f}'
                                      f'{t2_SN[p]:7.2f}')
                    
                else:
                    # Writing twin 2's info first (since it has the higher PI peak)
                    write_twins.write(f'\n{t2_gal_long[p]:7.3f}'
                                      f'{t2_gal_lat[p]:7.3f}'
                                      f'{t2_xpix[p]:7.0f}'
                                      f'{t2_ypix[p]:6.0f}'
                                      f'{t2_PI[p]:10.2f}'
                                      f'{t2_TI[p]:11.2f}'
                                      f'{t2_SN[p]:7.2f}')
                    write_twins.write(f'\n{t1_gal_long[p]:7.3f}'
                                      f'{t1_gal_lat[p]:7.3f}'
                                      f'{t1_xpix[p]:7.0f}'
                                      f'{t1_ypix[p]:6.0f}'
                                      f'{t1_PI[p]:10.2f}'
                                      f'{t1_TI[p]:11.2f}'
                                      f'{t1_SN[p]:7.2f}')
                    
    print(f'\nTwins sourcelist generated for mosaic {mosaic.upper()}!')



def Twin_classifying_multiple_mosaics(mosaics=None, filename="test", write_dat = True, 
                                      pausetime = 3, ):
    """
    This function goes through the mosaics indicated and identifies twins within them,
    and takes the user's classification as well. 
    
        It creates a csv with all the information about the twins. 
        This function also asks if any twin pairing were missed in the mosaics.
        
    Key Parameters:
        
        mosaics (string): 
            A list of mosaics the user wishes to go through.
            
            Default set to prompt for user input. 
        
        filename (string):
            The name of the csv file that is produced. 
        
        write_dat (bool):
            whether to write the .dat files for the twins or pairs found. Default: True
            
    Other Parameters:
        
        pausetime (int):
            The amount of time the user initially has to manipulate the plot of the 
            mosaic in both polarised intensity and Stokes I. 
    
    Returns:
        
        all_missing_twins (list):
            A list containing the mosaic, and the number of twins missing in the mosaic. 


"""
    import os
    import parameters_file as pf
    from pathlib import Path
    from directories import RM_out_dir, csv_dir, img_dir, backup_csv_dir
    plt.style.use('default')
    
    
    # Checking to make sure the directory exists.
    if os.path.isdir(csv_dir) == False:
        print("current csv directory: ", pf.csv_dir)
        print("Directory to store csv file does not exist, please correct in directories file. ")
        
        return np.nan

    
    # Setting the path of the csv file
    Path = csv_dir + filename+".csv"
    print("CSV path: ", Path)
    path_for_missing_twins = csv_dir +filename+"_missing_twins.csv"
                                       
    # writting backup paths
    main_csv_backup_path = backup_csv_dir+ filename+".csv"
    missing_twins_backup_path =backup_csv_dir + filename+"_missing_twins.csv"
    
    

    
    
    # Setting all the labels for the columns in the dataframe/csv
    labels = ["Mosaic", "Twin detected", "True twin detection",
                "Class", "True Class", "Polarized Intensity peak of twin 1 (Jy/beam)", "Polarized Intensity peak of twin 2 (Jy/beam)",
                "Total Intensity peak of twin 1 (Jy/beam)", "Total Intensity peak of twin 2 (Jy/beam)",
                
                "y coordinate of twin 1 in PI (pixel units)","x coordinate of twin 1 in PI (pixel units)","HWHM of twin 1 in PI (pixel units)",
                "y coordinate of twin 2 in PI (pixel units)","x coordinate of twin 2 in PI (pixel units)","HWHM of twin 2 in PI (pixel units)",
                
               "Galactic Latitude of twin 1 in PI (degrees)", "Galactic Longitude of twin 1 in PI (degrees)",
               "HWHM of twin 1 in PI (arcseconds)", "Galactic Latitude of twin 2 in PI (degrees)", 
               "Galactic Longitude of twin 2 in PI (degrees)","HWHM of twin 2 in PI (arcseconds)",
               
               "y coordinate of twin 1 in Stokes I (pixel units)","x coordinate of twin 1 in Stokes I (pixel units)","HWHM of twin 1 in Stokes I (pixel units)",
               "y coordinate of twin 2 in Stokes I (pixel units)","x coordinate of twin 2 in Stokes I (pixel units)","HWHM of twin 2 in Stokes I (pixel units)",
               
               "Galactic Latitude of twin 1 in Stokes I (degrees)", "Galactic Longitude of twin 1 in Stokes I (degrees)",
               "HWHM of twin 1 in Stokes I (arcseconds)", "Galactic Latitude of twin 2 in Stokes I (degrees)", 
               "Galactic Longitude of twin 2 in Stokes I (degrees)","HWHM of twin 2 in Stokes I (arcseconds)",
               "Signal to Noise of twin 1", "Signal to Noise of twin 2", 
               
               "Center y coor(pix)", "Center x coor(pix)", "Center b (deg)", "Center l (deg)", "distance btw lobes (pix)", "distance btw lobes (arcmin)",
               
               "Asso_w_AGN", "AGN_offset_from_center (arcmin)", "Name of AGN", "AGN l (deg)", "AGN b (deg)", "survey_AGN"]
    # associated_AGN, offset_AGN, name_AGN, AGN_long, AGN_lat, survey_AGN
    # Setting the initial number of missing twins to be 0
    all_missing_twins = []
    

    
    newfile = input("Is it this the start of a new csv file? \n"\
                       +"(enter 1 for yes and 0 for no): ")
    
        
    if newfile=="True" or newfile=="y" or newfile=="yes" or newfile=="1" or newfile=="":
        newfile=True
        header=True
        header_missing_twins = True
        mode_missing_twins = 'w'
        
    else:
        newfile=False
        header=False
        header_missing_twins = False
        mode_missing_twins = 'a'
        for p in [Path, main_csv_backup_path, missing_twins_backup_path, path_for_missing_twins]:
            with open(p, mode="a", newline='') as file:
                file.write("\n")
      
   
    
    
   
    # Getting the mosaics to examine if none were entered. 
    if mosaics==None:
        In = str(input("Please enter the mosaics you want to examine: "))
        # If multiple mosaics were entered, removing any spaces in the string and 
        #   spliting them where they are separated by a comma 
        if "," in In:
            In = In.replace(" ", "")
            mosaics = In.split(",")
            
    elif type(mosaics)==list:
        pass
    else:
        # If multiple mosaics were entered, removing any spaces in the string and 
        #   spliting them where they are separated by a comma 
        if "," in mosaics:
            mosaics = mosaics.replace(" ", "") #removes spaces
            mosaics = mosaics.split(",") # splits the mosaics
        else:
            mosaics=[str(mosaics)]
    
    while mosaics!= None:
        # looping through all the mosaics entered

        for mosaic in mosaics:
            
           
            mosaic = fc.try_path(mosaic, directory=img_dir)
            
            print("\nMosaic: ", mosaic.upper(), "\n")
         
            # Detecting the twins in polarized intensity and getting there coordinates,
            #   distance between them, the center point between them and the same thing but in galactic coordinates. 
            twin_list, distance_list, twin_centers, twinlist_galcoord, distlist_galcoord, twincentres_galcoord \
                = Potential_Twin_Finder(mosaic,  plot_individual_sources=True, 
                                                 Plot_twins=True,  plot_AGNs = False,
                                                return_gal_coord=2, PlotPI=False)
            
            # Giving the user a set amount of time to check if any twins were missed. 
            plt.pause(pausetime)    
            missing_twins_in_mosaic = input('''How many twins were missed in PI? (Enter "?" for more time): ''')
            
            # If the user just hits enter instead of entering a number take that to be 0. 
            if missing_twins_in_mosaic =="":
                missing_twins_in_mosaic = 0
            else: 
                missing_twins_in_mosaic = str(missing_twins_in_mosaic)
            # Going adding more time if the user hasn't determined if any twins
            #   are missing and needs to manipulate the graph. 
            
            while missing_twins_in_mosaic =="?":
                
                # geting the amount of time the user wants
                time_interval = input("How many more seconds do you want to look at the mosaic?")
                if time_interval!=0: # if the user gives an input
                    time_interval = int(time_interval)
                    
                    if time_interval >20: # checking if the user enters more than 20 seconds checking the amount entered
                        check = input("You entered a number greater than 20 seconds, is this correct (enter y or n)")
                        if check  =="n":
                            time_interval = input("How many more seconds do you want to look at the mosaic?")
                    #Giving the user graph manipulation time. 
                    plt.pause(int(time_interval))
                    # repeating the process
                    missing_twins_in_mosaic = input("How many twins were missed in PI? (Enter ? for more time): ")
                else:
                    # Getting the number of missed twins
                    missing_twins_in_mosaic = int(input("How many twins were missed in PI? "))
            
            # adding the number of twins missed in the mosaic, and which mosaic it was in
            all_missing_twins += [[mosaic.upper(), int(missing_twins_in_mosaic)]]
            
            # Getting all the data and classfications for the mosaics. 
            mosaic_dataset = twin_total_intensity_detector_and_classifier(mosaic,
                                                                          twin_centers, 
                                                                          twin_list=twin_list,
                                                                          distance=distance_list,
                                                                          plot_snapshots=True,  
                                                                          plot_AGNs = True,
                                                                          return_singular_list=True)
            
            df = pd.DataFrame([[mosaic.upper(), int(missing_twins_in_mosaic)]], columns = ["Mosaic", "Number of Twins"])
            df.to_csv(path_for_missing_twins, mode=mode_missing_twins, header = header_missing_twins, index = False)

            df.to_csv(missing_twins_backup_path, mode=mode_missing_twins, header = header_missing_twins, index = False)
            
            mode_missing_twins, header_missing_twins="a", False
            
            
            
            
            if mosaic_dataset != None:

                if newfile:
                    Mode="w"
                else:
                    Mode="a"
                # Adding that to the full data set
                # all_mosaics_dataset += mosaic_dataset.copy()
                DataFrame = pd.DataFrame(data=mosaic_dataset, columns = labels)
                # Determining the number of twins detected, this is done here so that 
                    # if none were detected, but script to write the data file does not run. 
                twins_detected = DataFrame["True twin detection"].tolist()
                
                
                
                #Creating or adding the data to the CSV file
                DataFrame.to_csv(path_or_buf=Path, mode=Mode, header=header, index=False, lineterminator="\n")
                DataFrame.to_csv(path_or_buf=main_csv_backup_path, mode=Mode, header=header, index=False, lineterminator="\n")
                
                
                header=False
                newfile=False

                if write_dat and np.sum(twins_detected) != 0:
                    
# =============================================================================
#               Code for writing the .dat file for it to work with Jo-Anne's code. 
#               Please see function above. 
# =============================================================================
                    write_dat_file(mosaic, mosaic_dataset,)
            
            
            plt.close("all")
            
                       
                    
                
        # Seeing if more mosaics should be examined. 
        new_mosaics = input("What other mosaics would you like to examine? \n Hit the enter key if you have finished. " )
        if new_mosaics == "":
            mosaics = None
        else:
            # mosaics=new_mosaics
            if "," in new_mosaics:
                In = In.replace(" ", "")
                mosaics = In.split(",")
            else:
                mosaics=[str(new_mosaics)]
    
   
    
    
    return all_missing_twins


list_all_mosaics = ['meq1', 'meq2', 'mer1', 'mer2', 'mes1', 'mes2', 'met1', 'met2',
                    'meu1', 'meu2', 'mev1', 'mev2', 'mew1', 'mew2', 'mex1', 'mex2',
                    'mey1', 'mey2', 'mez1', 'mez2', 'mst1', 'mst2', 'mu1', 'mu2',
                    'mv1', 'mv2', 'mw1', 'mw2', 'mx1', 'mx2', 'my1', 'my2', 'ma1',
                    'ma2', 'mb1', 'mb2', 'mc1', 'mc2', 'me1', 'me2','me3', 'me4', 
                    'me5', 'mf1', 'mf2', 'mf3', 'mf4', 'mf5',"mg0",'mg1', 'mg2', 'mg3',
                    'mg4', 'mg5', 'mh1', 'mh2', 'mh3', 'mh4', 'mh5', 'mij1', 
                    'mij2', 'mk1', 'mk2', 'mm1','mm2', 'mn1', 'mn2', 'mo1', 'mo2', 'mp1', 
                    'mp2', 'mq1', 'mq2','mr1', 'mr2', 'mej1', 'mej2', 'mek1', 
                    'mek2', 'mel1', 'mel2',"md1", "md2", "ml2", "ml1"]
list_mosaics_with_twins = ['meq1', 'meq2', 'mer1', 'mer2', 'mes1', 'mes2', 'met1', 'met2',
                    'meu1', 'meu2', 'mev1', 'mev2', 'mew1', 'mew2', 'mex1', 'mex2',
                    'mey1', 'mey2', 'mez1', 'mez2', 'mst1', 'mst2', 'mu1', 'mu2',
                    'mv1', 'mv2', 'mw1', 'mw2', 'mx1', 'mx2', 'my1', 'my2', 'ma1',
                    'ma2', 'mb1', 'mb2', 'mc1', 'mc2', 'me1', 'me2','me3', 'me4', 
                    'me5', 'mf1', 'mf2', 'mf3', 'mf4', 'mf5',"mg0",'mg1', 'mg2', 'mg3',
                    'mg4', 'mg5', 'mh1', 'mh2', 'mh3', 'mh4', 'mh5', 'mij1', 
                    'mij2', 'mk1', 'mk2', 'mm1','mm2', 'mn1', 'mn2', 'mo1', 'mo2', 'mp1', 
                    'mp2', 'mq1', 'mq2','mr1', 'mr2', 'mej1', 'mej2', 'mek1', 
                    'mek2', 'mel1', 'mel2',]




mosaics_to_go_through = [ "mer1", "md1"]



missing = Twin_classifying_multiple_mosaics(mosaics=mosaics_to_go_through, filename="test", write_dat=False)






def write_dat_from_csv(csv_dir=None):
    """This function writes .dat files from a csv of pair or twin sources from 
    pair finder script. 
    
    Input Parameters:
        csv_dir: the path to the csv
        
    Final output:
        writes .dat files for the pairs in each mosaic to wherever the 
        RM_out_dir defined in the parameters file. """
    if csv_dir == None:
        csv_dir = """/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Codes/TwinFinding/FinalRound/Final_Twin_List_twins_only.csv"""

    
    all_twins_df = pd.read_csv(csv_dir, index_col="n")
    
    mosaics = all_twins_df["Mosaic"].unique()
    
    for m in mosaics:
        mosaic_df = all_twins_df[all_twins_df["Mosaic"]==m]
        
        
        mosaic_df_list = mosaic_df.values.tolist()
       
        write_dat_file(m, mosaic_df_list)
            

