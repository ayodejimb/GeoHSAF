import os
def count_nifti_files(root_dir):
    nifti_count = 0 

    for dirpath, _, filenames in os.walk(root_dir):
        nifti_files = [f for f in filenames if f.lower().endswith('.nii')]
        nifti_count += len(nifti_files)
        if nifti_files:
            print(f"Found {len(nifti_files)} NIfTI files in: {dirpath}")

    print(f"\nTotal NIfTI files found: {nifti_count}")

root_directory = r"C:\.......\ADNI\AD_Subjects" 
count_nifti_files(root_directory)
