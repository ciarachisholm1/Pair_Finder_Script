# pair_finder_draft
The code I created to find closely spaced pairs in the CGPS polarised intensity images. This code was created before I knew the previously determined names for physical and random pairs, so many times I refer to the pairs as "twins". 


I will update at a later date a modified version of the code intended for future use by other users. This code is being uploaded for transperency purposes. 

To run the code as is, download code and place into a directory. When running the code, ensure that the parent directory is to the folder containing the `directories.py`, `Functions.py`, and `parameters_file.py`.  

## Setup instructions
### Data setup

The first step is to modify the directories file. The user must update the image directory (named img_dir) to where the FITS files are located on their device. Next is to update csv_dir, where the csv containing all the found pairs should be stored. As an option, the user may choose to modify where the backup csv files are stored at this stage as well. 

If the user wishes to write dat for the sources detected to utilize the Rotation Measure Analysis Program (RMAP), the path for the dat files must be written. 

The images created during the process are stored, if the user enables the feature, via the paths
* pot_twin_img_dir, where the whole mosaic with all the detected sources are stored
* cutout_img_dir, where the cutouts and the 3D maps of each pair are stored

### Parameters setup
The Parameters file contains all the changable parameters for the program. These include minimum and maximum size of the sources to detect, plotting parameters, etc. They can easily be changed by the user. 

## Running the programs
### Automated pair finding

To run the automated program, open the file `Twinfinding_fully_automated.py` and run the function: Twin_classifying_multiple_mosaics, found at the end of the script. The user must enter the mosaics to find the pairs within, and the user should enter filename of the csv to store the pair information in, and whether it is a new csv (i.e. create a new csv file or replace an existing one) or append to an existing csv file. The program will print which mosaic it is examining and some of the information of the pairs it found. 
Note: This function does not automatically write dat files of the pairs. If the user wishes to write dat files, go to line 2636 and change the variable "write_dat" from False to True. 

### User input pair finding

This program finds the pairs in the mosaics but verifies with the user whether or not the pairs detected are true pairs and if any are missing. 
To run this program, open the file `Twinfinding_classification_latest.py` and run the function "Twin_classifying_multiple_mosaics". The user must enter the mosacis to examine, if none are entered the user prompted in the command line. The user should also enter the filename of the csv to created to store the pair information. The user is as asked by the program whether to write a new csv (or overwrite an old one) or to append to an existing csv file. 
The program will then produce an image of the mosaic in both polarised intensity (PI) and Stokes I (SI) with the detected pairs and solo sources highlighted. The program then asks if there are any pairs missing, if there are none enter 0. If more time to manipulate the graphs is required to determine if any are missing, enter a "?". 
The program will then produce cutouts of the detected pairs in PI and SI and in both 2D and 3D. If the pair is a true pair, it will be given a classification greater than 500. If it fails any of the tests (See Ciara Chisholm's MSc thesis for details on the classification scheme). If the classification is correct the user can simply hit enter to move onto the next pair, else enter the proper classification number. 
This repeats for all pairs in the mosaic, the program then adds the information to the csv. The csv is updated after all the pairs in a mosaic have been examined so if any issues occur during the process, the user does not need to re-examine every mosaic. 
Once all the pairs have been examine in the mosaic, the program repeats the steps for the next mosaic in the lst. 

A csv file containing all the pair information is found by the path: csv_dir + filename, filename is entered by the user. 
