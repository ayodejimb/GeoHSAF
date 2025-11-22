from sklearn.model_selection import StratifiedKFold
import numpy as np
import os, shutil
from collections import defaultdict

base_path = r'C:\.....\ADNI\After_matching'
condis = ['AD', 'CN']   # for binary
# condis = ['AD', 'CN', 'MCI'] # for multi-class

# === Collect metadata ===
metadata = []  # (condi, file, timepoint, subject_id)
for condi in condis:
    folder_path = os.path.join(base_path, condi)
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt') and filename.startswith('matched_'):
            base = filename[len('matched_'):].replace('.txt', '')
            if '_' in base:
                *subject_parts, timepoint = base.split('_')
                subject_id = '_'.join(subject_parts)
                metadata.append((condi, filename, timepoint, subject_id))

# subjects with baseline
subject_timepoints = defaultdict(set)
for _, _, timepoint, subject_id in metadata:
    subject_timepoints[subject_id].add(timepoint)
subjects_with_bl = {subj for subj, tps in subject_timepoints.items() if 'bl' in tps}
metadata = [m for m in metadata if m[3] in subjects_with_bl]

# Subject counts
subject_time_counts = defaultdict(int)
for _, _, _, subject_id in metadata:
    subject_time_counts[subject_id] += 1
elig_t_subj = [subj for subj, count in subject_time_counts.items() if count >= 3]
inelig_t_subj = [subj for subj, count in subject_time_counts.items() if count < 3]
subject_to_group = {m[3]: m[0] for m in metadata}
elig_lab = [subject_to_group[subj] for subj in elig_t_subj]
# === 5-fold stratified split ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
def copy_files(metadata_split, split_name, fold_idx):
    for condi, filename, _, _ in metadata_split:
        src_path = os.path.join(base_path, condi, filename)
        dst_dir = os.path.join(new_base_path, f"fold{fold_idx}", condi+split_name)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, filename)
        shutil.copy2(src_path, dst_path)

new_base_path = r'C:\Users\.....\After_matching\Splitted'
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.array(elig_t_subj),
                                                           np.array(elig_lab)), start=1):

    train_subjects = set(np.array(elig_t_subj)[train_idx]) | set(inelig_t_subj)
    test_subjects  = set(np.array(elig_t_subj)[test_idx])
    metadata_train = [m for m in metadata if m[3] in train_subjects]
    metadata_test  = [m for m in metadata if m[3] in test_subjects]

    print(f"\nFold {fold_idx}")
    print(f"  Train subjects: {len(train_subjects)}, Test subjects: {len(test_subjects)}")
    print("  Train group distribution:", np.unique([m[0] for m in metadata_train], return_counts=True))
    print("  Test group distribution:",  np.unique([m[0] for m in metadata_test], return_counts=True))

    # Copy files for this fold
    copy_files(metadata_train, "_train", fold_idx)
    copy_files(metadata_test, "_test", fold_idx)

print("5-Fold Stratified Subject-wise Splitting Completed.")

