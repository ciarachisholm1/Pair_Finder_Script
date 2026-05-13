#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:44:09 2024

@author: ciarachisholm
"""




import os 
# This gets the path to the current directory. This is so it automatically looks for the files in the correct place. 
# If this is being run on the desktop computer, it will return /home/ciara.chisholm1/OneDrive/Ciara's Research Cubby/
# If this is being run on the laptop, it should return /Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby
onedrive_path = os.path.dirname(os.path.realpath(__file__))[:-5]


if onedrive_path[1:5] =="home":
    img_dir = "/scratch/CGPSData/CGPS2025/"
elif onedrive_path[1:5] =="User":
    img_dir = "/Users/ciarachisholm/Desktop/Research/CGPS2025/"
else:
    print("WARNING: Image directory not configured for the current device. Please add image directory option to the directories file. "+\
          "The data should ideally be stored locally to avoid a bottle neck when loading the datafiles.")


# print(onedrive_path)
# csv_dir = "/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Codes/rm_test1/"
# /Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara\'s\ Research\ Cubby/Codes/TwinFinding/min_threshold

twin_algorithm_folder = "ext_FITS_fixed"#"add_StoN_condition"

csv_dir = onedrive_path +"Codes/TwinFinding/" +twin_algorithm_folder+"/"

backup_csv_dir = onedrive_path +"Codes/TwinFinding/"+twin_algorithm_folder+"/Backup_files/"

cm_dir = onedrive_path +"Figures/Twinfinding_algorithm/" +twin_algorithm_folder+"/Confusion_Matrices/"

# RM_out_dir= onedrive_path +"Codes/rmap-main1/Data_StoN_condition/output_data/"
RM_out_dir= onedrive_path + "Codes/rmap-main1/ext_FITS_fixed/output_data/" #"Codes/rmap-main1/One_TI_src_pairs/output_data/"

cam_cat_dir = onedrive_path +"Codes/RMTable/VanEck2021.fits"

man_twin_csv_dir = onedrive_path +"Codes/Manual twin finding/twins_found.csv"

all_RMs_dir = onedrive_path +"Codes/rmap-main1/misfit_pairs/"

cutout_img_dir = onedrive_path +"Codes/TwinFinding/"+twin_algorithm_folder+"/cutouts/"

pot_twin_img_dir = csv_dir +"Potential_twin_plots/"

# true_twin_csv_dir = onedrive_path + "Codes/TwinFinding/true_twins_list.csv"
true_twin_csv_dir = onedrive_path + "Codes/TwinFinding/true_twins_list_my_AGN_asso_new_simbad.csv"
 
rm_fig_folder = "StoN_condition"

rm_fig_dir = onedrive_path +"""Figures/RM_plots/"""+rm_fig_folder+ "/RMs/"

asso_plot_dir = csv_dir +"Associated_plots"