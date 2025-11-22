import os
import subprocess

def find_dicom_and_nifti_folders(root_dir):
    failed_conversions = 0
    
    for dirpath, _, filenames in os.walk(root_dir):
        dicom_files = [f for f in filenames if f.lower().endswith('.dcm')]
        nifti_files = [f for f in filenames if f.lower().endswith('.nii')]

        if nifti_files:
            print(f"Skipping {dirpath} (already contains NIfTI files)")
        elif dicom_files:
            print(f"Converting DICOM to NIfTI in {dirpath}...")
            success = convert_dicom_to_nifti(dirpath)
            if not success:
                failed_conversions += 1
    
    print(f"Total failed conversions: {failed_conversions}")

def convert_dicom_to_nifti(dicom_folder):
    output_folder = dicom_folder  # Save NIfTI in the same folder
    
    try:
        subprocess.run(["dcm2niix", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Error: dcm2niix is not installed or not found.")
        return False

    try:
        subprocess.run(["dcm2niix", "-o", output_folder, dicom_folder], check=True)
        return True  # Successful conversion
    except subprocess.CalledProcessError:
        print(f"Failed to convert DICOM in {dicom_folder}")
        return False

# Set your main directory here
root_directory = r"C:\.....\AIBL\AD"
# root_directory = r"C:\.....\AIBL\CN"
# root_directory = r"C:\.....\AIBL\MCI"

# Start the conversion process
find_dicom_and_nifti_folders(root_directory)
