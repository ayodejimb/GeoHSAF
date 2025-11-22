import os
import shutil
import pandas as pd

# Define paths
main_folders = [r"C:\......\OASIS\OAS2_RAW_PART1\OAS2_RAW_PART1", r"C:\.......\OASIS\OAS2_RAW_PART2\OAS2_RAW_PART2"]  # Update with actual paths (downloaded images from OASIS)
output_dir = r"C:\......\OASIS\OAS2_PREPROCESSED" 

# Create output folders
demented_folder = os.path.join(output_dir, "demented")
non_demented_folder = os.path.join(output_dir, "non_demented")

os.makedirs(demented_folder, exist_ok=True)
os.makedirs(non_demented_folder, exist_ok=True)

# Load Excel file
excel_path = r"C:\......\oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"  # Update with Excel file from OASIS
df = pd.read_excel(excel_path)

# Iterate over rows
for _, row in df.iterrows():
    mri_id = row["MRI ID"]
    group = row["Group"]

    print(group, mri_id)

    # Extract subject ID and scan ID
    parts = mri_id.split("_")
    subject_id = parts[1].lstrip("0")
    scan_id = parts[2][-1] 

    # Determine target folder
    if group == "Demented":
        target_folder = demented_folder
    elif group == "Nondemented":
        target_folder = non_demented_folder
    else:
        continue 

    # Search for the correct file
    found = False
    for main_folder in main_folders:
        search_path = os.path.join(main_folder, mri_id)  # Unique folder name
        raw_folder = os.path.join(search_path, "raw")
        nii_file_path = os.path.join(raw_folder, "mpr-1.nii")

        if os.path.exists(nii_file_path):
            # Define new filename
            new_filename = f"S{subject_id}_{scan_id}_mpr-1.nii"
            destination_path = os.path.join(target_folder, new_filename)

            # Copy file
            shutil.copy(nii_file_path, destination_path)
            print(f"Copied: {nii_file_path} -> {destination_path}")
            found = True
            break  # Stop searching once found

    if not found:
        print(f"File not found for MRI ID: {mri_id}")

print("Processing complete.")



