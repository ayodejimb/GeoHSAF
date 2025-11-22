import os
import shutil
import re
import pandas as pd

condi = 'AD'
# condi= 'CN'

csv_file = r'C:\....\OASIS\After_matching\oasis_longitudinal_demographics-8d83e569fa2e2d30.csv'
txt_folder = r'C:\....\OASIS\After_matching\{}_unrenamed'.format(condi)
output_folder = r'C:\....\OASIS\After_matching\{}'.format(condi)
os.makedirs(output_folder, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_file)

# Helper: extract numeric ID e.g., "OAS2_0001_MR1" => 1
def extract_csv_id(mri_id):
    match = re.search(r'OAS2_(\d+)_', mri_id)
    return int(match.group(1)) if match else None

# Preprocess CSV to map subject ID to ages per visit
df['subject_id'] = df['MRI ID'].apply(extract_csv_id)
visit_age_map = {}
for _, row in df.iterrows():
    sid = row['subject_id']
    visit = row['Visit']
    age = row['Age']
    if pd.notnull(sid):
        visit_age_map.setdefault(sid, {})[visit] = age

# Build age repetition count per subject
age_seen_count = {sid: {} for sid in visit_age_map}

for fname in os.listdir(txt_folder):
    if fname.endswith('.txt'):
        match = re.match(r'matched_S(\d+)_([0-9]+)_', fname)
        if not match:
            continue

        subject_num = int(match.group(1))
        visit_num = int(match.group(2))

        if subject_num not in visit_age_map:
            print(f"Subject {subject_num} not found in CSV, skipping.")
            continue
        visits = visit_age_map[subject_num]

        if 1 not in visits or visit_num not in visits:
            print(f"Missing visit age for subject {subject_num}, skipping.")
            continue

        base_age = visits[1]
        visit_age = visits[visit_num]

        # Count age occurrences
        age_seen = age_seen_count[subject_num]
        age_seen[visit_age] = age_seen.get(visit_age, 0) + 1
        repeat_offset = (age_seen[visit_age] - 1) * 6 

        # Determine new filename
        if visit_num == 1:
            new_name = f"matched_S{subject_num}_bl.txt"
        else:
            age_diff_years = visit_age - base_age
            months = int(round(age_diff_years * 12)) + repeat_offset
            if months == 6:
                month_str = "06"
            else:
                month_str = str(months)
            new_name = f"matched_S{subject_num}_m{month_str}.txt"

        src_path = os.path.join(txt_folder, fname)
        dst_path = os.path.join(output_folder, new_name)
        shutil.copyfile(src_path, dst_path)
        print(f"Copied {fname} to {new_name}")
