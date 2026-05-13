#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:18:16 2024

@author: ciarachisholm
"""
from numpy import sqrt, log
from directories import img_dir, csv_dir

# =============================================================================
# Main changes from round 4:
#    • Window width changed from 34 to 28
#    • min radius changed from 0.5*beam_radius to 0.75*beam_radius
# =============================================================================

############################## Image Directory ###############################

# img_dir = "/Users/ciarachisholm/Desktop/Research/CGPS2012/"
img_dir = img_dir
csv_dir = csv_dir
# print("img_dir: ", img_dir)

######################## General Plot Parameters #######################

num_of_pixels_btw_ticks = 150 #The number of ticks between pixels for full mosaic plots
DPI =100 
beam_radius=1.61 # =(58"/(17.9578830553509762" per pixel))/2, 58" is the FWHM of the beam in it's smallest form. 
beam_sigma = (3.22)/(2*sqrt(2*log(2)))
# Note this is the gaussian beam width. The untapered beam width is 49". 
# csv_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Codes/TwinFinding/Detecting_text_files/"

######################## Polarised Intensity Parameters #######################

PI_VMIN = 0.0 # minimum value in the colormap
# PI_VMAX = 0.006 # maximum value in the colormap
PI_VMAX = 0.003 # maximum value in the colormap
PI_num_of_pixels_btw_ticks = 3

########################## Total Intensity Parameters #########################

TI_VMIN = 0.0
TI_VMAX = 0.03
# TI_VMAX = 0.3

######################### LoG Blob Detection Parameters #######################

# max_sigma = 5
# min_sigma = 0.5*beam_radius
# following changed by Ciara Chisholm on Nov 12
max_sigma=4*beam_sigma
min_sigma = 0.5*beam_sigma
num_sigmas = 40
threshold = 1.5/2000# The PI threshold is 1.5 mJy/beam and the peak in scale space
overlap_LoG =0.5# was 0.8

    ######## cut out parameters #######

mosaic_edge_weight_threshold = 0.15 # for the singular field edges

#mosaic_overlap_width is automatically set to 109. The full overlap is 112, and 
#   this gives some room to detect twins on the edge of the overlap region,
#   but not outside of it.
mosaic_overlap_width = 109 

#################### Potential (PI) twin finder parameters ####################

max_dist_btw_sources=27# pixels was initially 18, then changed to 15.5, now changed to 27 pixels after catalogue analysis
# max_radius=2*beam_radius # was 1.25*beam_radius, was changed to 2 beam radius in accordance with the Seyfret galaxy and JB thesis.
max_radius = 5 # was the above, now is based on the max size of a compact source in JB thesis which is 3' in diameter page 53.
min_radius=0.75*(beam_radius)
min_dist_btw_sources = beam_radius*3#round(beam_radius*2) # just over a beam width to be able to distinguish sources when fittting
ratio_threshold_PI = 10# the max ratio the PI peak values of the twins can be to be considered twins, was 6



    ######## plot parameters ########
PTF_axis_font_size = 25
PTF_titlefontsize=30
NaNcolor='white'
mask_alpha=0.5
# Setting the sources found to not plot on an separate figure. 
separate_individual_source_plot=False
# Setting the patch color for the legend of the Polarised and total intensity plot
# Patchcolor_edge='gold'
Patchcolor_edge='yellow'
Patchcolor_overlap = "silver"
# Setting whether to print the coordinates
print_coordinates=False
radius_scale=1# Not scaling the circles ploted


artifact_window_dist = 25 #pixels
# To completely remove the artifact in MEJ1, it had to be set to 40
max_num_of_sources_in_artifact_window = 5# This includes that identified twin sources, so no more than 3 pairs in the window. 
ratio_TI_to_PI = 4/3 # was 3/2 based on Cam's cat, changed to 4/3 to fit the theoritical limit. 
min_StoN_ratio = 5


################### (TI) twin finder and class. parameters ####################


max_offset = round(3*beam_radius) # the maximum offset between the PI sources and TI sources, was 7 

# offset_tolarance = 3 # No longer used. 
num_btw_ticks_snapshots=14# was 16
snapshot_length=42# !!Note: must be a even number. This is length and height of the TI plot snapshot cut outs
initial_graph_time=3# seconds

# Setting a max and min radius for a twin source in Stokes I
# max_TI_radius=2.4*beam_radius
max_TI_radius = 5 # Was the above changed to match Jo-Anne's thesis 
min_TI_radius=0.75*beam_radius# Smae as for PI

ratio_threshold_TI = 10## the max ratio the TI peak values of the twins can be to be considered twins





















