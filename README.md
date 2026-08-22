# pair_finder_draft
The code I created to find closely spaced pairs in the CGPS. This code was created before I knew the previously determined names for physical and random pairs, so many times I refer to the pairs as "twins". 


I will update at a later date a modified version of the code intended for future use by other users. This code is being uploaded for transperency purposes. 

To run the code as is, download code and place into a directory. 

### Setup instructions
#### Data step up

The first step is to modify the directories file. The user must update the image directory (named img_dir) to where the FITS files are located on their device. Next is to update csv_dir, where the csv containing all the found pairs should be stored. As an option, the user may choose to modify where the backup csv files are stored at this stage as well. 

If the user wishes to utilize the confusion matrix creater function, then the path for the confusion matrix must also be updated. 

If the user wishes to write dat for the sources detected to utilize the Rotation Measure Analysis Program (RMAP), the path for the dat files must be written. 

The images created during the process are stored, if the user enables the feature, via the paths
* pot_twin_img_dir, where the whole mosaic with all the detected sources are stored
* cutout_img_dir, where the cutouts and the 3D maps of each pair are stored

