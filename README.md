## **GeoHSAF: Geometric Hippocampus Shape Analysis Framework for Longitudinal Alzheimer's Disease Classification**

# Dataset and Codes

## Dataset
We use three public longitudinal AD datasets: ADNI, OASIS and AIBL. In the 'Datasets' folder, we include the scripts to download and organize the MR images after downloading. The scripts should be run in the following order:
- #### ADNI
  **⚡(1)** Run `get_category_to_csv.py` file to extract subjects as .csv  **⚡(2)** Run `get_ptid_vscode_and_uid.py` file to get the subjects ID for downloading the MRI scans from LONI **⚡(3)** After downloading the MRI scans, run `convert_dcm_to_nii.py` to convert dicom files to .nii **⚡(4)** Run `count_nifti.py` to ensure all files are correctly converted **⚡(5)** Run `preprocessed_to_timepoints.py` to organize files for segmentation. <br>
- #### OASIS
  Download OASIS-2 from [here](https://sites.wustl.edu/oasisbrains/datasets/). Then Run **⚡(1)** the file `convert_files_to_nii.py` to convert files to .nii and **⚡(2)** the file `extract_time_points.py` to organize for segmentation.
- #### AIBL
   **⚡(1)** Run `get_subj_id_for_download.py` file to obtain subjects ID for downloading from LONI. **⚡(2)** After downloading the MRI scans, run `convert_dcm_to_nii.py` to convert files to .nii **⚡(3)** `remove_intermidiary_folder.py` for folder consistency. **⚡(4)** Run AIBL_data_preprocessed.py to organize files for segmentation. 

## Segmentation
We use the FSL toolbox for the hippocampus segmentation. See procedures [here](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/FIRST(2f)StepByStep.html).

## Packages and Dependencies
- Create a separate environment and install the packages and dependencies in the `environment.yml` file using conda - `conda env create -f environment.yml`. For pip, first create an enviroment using Python, activate the enviroment and run `pip install -r requirements.txt`. Note that with pip, some of the dependencies and packages may not be installed and you have to install them yourself. 

## Removal of Rigid Transformations, and Surface Matching
All files for this are in the `alignment_and_matching` folder
**⚡(1)** Run `from_vtk_to_off.py` to pass from .vtks to .off  **⚡(2)** Run `normalization.py` and `alignment.py` accordingly to remove rigid transformations and then **⚡(3)** `non_rigid_matching.py` and `extraction_of_matches_code.py` accordingly for surface matching. The DGCNN folder needed for running the `non_rigid_matching.py` can be downloaded from [here](https://drive.google.com/file/d/1nlFtehis340g-vOrKmXxz1KV1Uu8Z253/view?usp=drive_link) 

For the OASIS dataset, run **⚡(4)** the file `rename_files_with_timepoints.py` to rename the files to include months of the scans in the file names. 

## Mean and Tangent Projections
**⚡(1)** Run the file `Subj_wise_strat_splitting_five_folds.py` for subject-wise five fold split. Then in the `mean_and_tangent_projections` folder, run **⚡(a)** the file `Karcher_mean_and_tangent_proj_for_train.py` to compute the mean on train set and tangent projection for train set. And then run **⚡(b)** `Tangent_proj_for_test.py` for tangent projection for test set. 

## Interpolation
In the `interpolation` folder, run accordingly **⚡(1)**  the file `pga.py` for PGA,  and then **⚡(2)** the file `gpr.py` for interpolation.

## Binary Classification
In the `binary_classification` folder, run the file `classification.py` for AD vs. NC classification.

## Multi-Class Classification
In the `multi_class_classification` folder, run the file `tr_classification_datasetname` for ADNI or AIBL as the case may be.

## Ablation Study Files
Files are available in the `ablation_study` folder. 

## License
This code and models are available for non-commercial scientific research purposes as defined in the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party codes are subject to their respective licenses.

