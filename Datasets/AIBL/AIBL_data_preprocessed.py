import pandas as pd
from collections import defaultdict
import re
import os
import shutil

df = pd.read_csv(r'C:\.....\AIBL\AD_from_LONI_based_on_RID.csv')  # .csv file of downloaded data from LONI
data_root = r"C:\.....\AIB\AD"    # path of the nifti files
output_folder = r'C:\...\AIBL_AD_Organized'
os.makedirs(output_folder, exist_ok=True)

# Build subjects and their scans into a dictionary
subject_visits = defaultdict(list)

def convert_visit(visit):
    if isinstance(visit, str):
        if 'Baseline' in visit:
            return 'bl'
        match = re.search(r'(\d+)\s*Month', visit)
        if match:
            return f"m{match.group(1)}"
    return visit  # fallback in case it doesn't match

for _, row in df.iterrows():
    subject_id = row['Subject ID']
    visit = convert_visit(row['Visit'])
    subject_visits[subject_id].append(visit)

subject_visits = dict(subject_visits)
subject_visits = {k: list(dict.fromkeys(v)) for k, v in subject_visits.items()}

# print(subject_visits)
# print(f"Total number of subjects: {len(subject_visits.keys())}")
# total_scans = sum(len(visits) for visits in subject_visits.values())
# print(f"Total number of scans: {total_scans}")

# Organization of the files in time (for CN or MCI, don't forget to change accordingly above)
for subject_id, visits in subject_visits.items():
    subject_path = os.path.join(data_root, str(subject_id))

    if not os.path.isdir(subject_path):
        print(f"Skipping {subject_id}: folder does not exist.")
        continue

    visit_folders = [f for f in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, f))]

    if len(visit_folders) != len(visits):
        print(f"Mismatch for {subject_id}: {len(visit_folders)} folders vs {len(visits)} visits")
        continue

    # Sort visit folders and visits to keep consistent mapping
    visit_folders.sort()
    visits.sort()

    # print(visit_folders, visits)

    for visit_folder, visit_code in zip(visit_folders, visits):
        full_visit_path = os.path.join(subject_path, visit_folder)

        # Walk the directory tree to find the .nii file
        nii_found = False
        for root, dirs, files in os.walk(full_visit_path):
            for file in files:
                if file.endswith('.nii') or file.endswith('.nii.gz'):
                    nii_found = True
                    src_path = os.path.join(root, file)
                    dest_filename = f"{subject_id}_{visit_code}.nii"
                    dest_path = os.path.join(output_folder, dest_filename)

                    shutil.copy2(src_path, dest_path)
                    print(f"Copied {src_path} -> {dest_path}")
                    break
            if nii_found:
                break
        if not nii_found:
            print(f"No .nii file found for {subject_id} at visit {visit_code}")

print("Done.")
