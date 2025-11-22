import pandas as pd
import os
import shutil

file_path = r"C:\.....\ADNI\AD_only.csv"

df = pd.read_csv(file_path, low_memory=False)

ptid_dict = {}

# Iterate through the DataFrame
for _, row in df.iterrows():
    ptid = row["PTID"]
    viscode = row["VISCODE"]
    imageuid = row["IMAGEUID"]
    
    if ptid not in ptid_dict:
        ptid_dict[ptid] = [[], []]
    
    # Append VISCODE and IMAGEUID to the respective lists
    imageuid = int(imageuid) if pd.notna(imageuid) else imageuid
    ptid_dict[ptid][0].append(viscode)
    ptid_dict[ptid][1].append(imageuid)

# print(ptid_dict)
for ptid, values in ptid_dict.items():
    viscodes, imageuids = values 
    
    # Create a new filtered list excluding NaN imageuids
    cleaned_viscodes, cleaned_imageuids = zip(*[
        (viscode, imageuid) for viscode, imageuid in zip(viscodes, imageuids) if pd.notna(imageuid)
    ]) if any(pd.notna(imageuid) for imageuid in imageuids) else ([], [])

    # Update the dictionary
    ptid_dict[ptid] = [list(cleaned_viscodes), list(cleaned_imageuids)]

ptid_dict = {key: value for key, value in ptid_dict.items() if value[0]}
# print(ptid_dict)
# print(len(ptid_dict.keys()))

# *** organizing to time_points here *****
def copy_and_rename_nii(main_folder, destination_folder, mapping):
    for subject_folder in os.listdir(main_folder):
        subject_path = os.path.join(main_folder, subject_folder)
        
        if not os.path.isdir(subject_path) or subject_folder not in mapping:
            continue  # Skip if not a directory or not in the dictionary
        
        # Recursively find the last-level folders with 'I' prefix
        for root, dirs, files in os.walk(subject_path):
            for dir_name in dirs:
                if dir_name.startswith('I'):
                    number = dir_name[1:]  # Remove 'I' to get the number
                    
                    if number.isdigit() and int(number) in mapping[subject_folder][1]:
                        idx = mapping[subject_folder][1].index(int(number))
                        nii_label = mapping[subject_folder][0][idx]  # Get corresponding label
                        
                        nii_source_folder = os.path.join(root, dir_name)
                        nii_filename = f"{subject_folder}_{nii_label}.nii"
                        nii_dest_path = os.path.join(destination_folder, nii_filename)
                        
                        # Find .nii file inside the folder and copy
                        for file in os.listdir(nii_source_folder):
                            if file.endswith(".nii"):
                                nii_source_file = os.path.join(nii_source_folder, file)
                                shutil.copy(nii_source_file, nii_dest_path)
                                print(f"Copied: {nii_source_file} -> {nii_dest_path}")
                                break  # Copy only the first .nii file found

# *** AD for ex... ****
main_folder = r"C:\......\ADNI\AD" # folder of the downloaded Nifti files
destination_folder = r"C:\.....\ADNI_PREPROCESSED\AD"
copy_and_rename_nii(main_folder, destination_folder, ptid_dict)