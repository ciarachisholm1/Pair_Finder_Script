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
          "The data should ideally be stored locally to avoid a slow down when loading the datafiles.")




twin_algorithm_folder = "ext_FITS_fixed"#"add_StoN_condition"

csv_dir = onedrive_path +"Codes/TwinFinding/" +twin_algorithm_folder+"/"

backup_csv_dir = onedrive_path +"Codes/TwinFinding/"+twin_algorithm_folder+"/Backup_files/"

RM_out_dir= onedrive_path + "Codes/rmap-main1/ext_FITS_fixed/output_data/" 

cutout_img_dir = onedrive_path +"Codes/TwinFinding/"+twin_algorithm_folder+"/cutouts/"

pot_twin_img_dir = csv_dir +"Potential_twin_plots/"

