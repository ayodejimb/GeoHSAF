import torch
import gpytorch
import os
import pickle
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

splits_root = r"C:\....\ADNI\After_matching\Splitted"
folds = sorted([f for f in os.listdir(splits_root) if f.startswith("fold")])

for fold in folds:
    print(f"------ NOW RUNNING FOR ------{fold}")
    base_path = r"C:\....\ADNI\After_matching\Splitted\{}".format(fold)
    def extract_time_map_global(base_path, condis):
        time_labels = set()
        for condi in condis:
            data_folder = os.path.join(base_path, f"{condi}_PCA_tangent_data")
            for file in os.listdir(data_folder):
                if not file.endswith('.pkl'):
                    continue
                with open(os.path.join(data_folder, file), 'rb') as f:
                    data = pickle.load(f)
                    time_labels.update(data.keys())

        def convert(label):
            if label == 'bl':
                return 1
            elif label.startswith('m') and label[1:].isdigit():
                return int(label[1:])
            else:
                return None

        valid_labels = [label for label in time_labels if convert(label) is not None]
        time_map = {label: convert(label) for label in sorted(valid_labels, key=convert)}
        return time_map

    global_time_map = extract_time_map_global(base_path, ['CN_train', 'AD_train'])
    inverse_time_map = {v: k for k, v in global_time_map.items()}
    # print("Global time map:", global_time_map)
    print("Global time map:", len(global_time_map.keys()))

    # ---------------- Interpolation with neigboring scans ----------------
    condis = ['CN_train', 'AD_train', 'CN_test', 'AD_test']
    for condi in condis:
        data_folder = os.path.join(base_path, f"{condi}_PCA_tangent_data")
        save_folder = os.path.join(base_path, f"{condi}_false_interpolated_PCA_tangent_data")
        os.makedirs(save_folder, exist_ok=True)

        for file in tqdm(os.listdir(data_folder), desc=f"Processing {condi}"):
            if not file.endswith('.pkl'):
                continue
            subject_key = file.split('_pca')[0]

            with open(os.path.join(data_folder, file), 'rb') as f:
                data = pickle.load(f)
            
            # Convert subject times into integers
            subject_times = {global_time_map[k]: v for k, v in data.items() if k in global_time_map}

            if len(subject_times) < 2:

                # we save, nothing to interpolate (interpolated added only to the name for consistency)
                save_path = os.path.join(save_folder, f"{subject_key}_interpolated.pkl")
                with open(save_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f"Saved subject {subject_key} ({condi}) to {fold}")

                continue

            all_times_sorted = sorted(subject_times.keys())
            missing_times = set(global_time_map.values()) - set(all_times_sorted)

            # if exactly 2 scans 
            if len(all_times_sorted) == 2:
                t0, t1 = all_times_sorted[0], all_times_sorted[-1]
                in_between = [t for t in range(t0 + 1, t1) if t in global_time_map.values()]
                
                # no missing between, skip
                if not any(t in missing_times for t in in_between):

                    # we save, nothing to interpolate (interpolated added only to the name for consistency)
                    save_path = os.path.join(save_folder, f"{subject_key}_interpolated.pkl")
                    with open(save_path, 'wb') as f:
                        pickle.dump(data, f)
                    print(f"Saved subject {subject_key} ({condi}) to {fold}")

                    continue

                else:
                    # get missing time points and fill these with neighboring scans
                    for t in in_between:
                        if t in missing_times:
                            if abs(t - t0) <= abs(t1 - t):
                                subject_times[t] = subject_times[t0]
                            else:
                                subject_times[t] = subject_times[t1]

            if len(all_times_sorted) > 2:
                middle = all_times_sorted[1:-1]

                if len(middle) == 1:
                    # Only one middle scan
                    t = middle[0]

                    # fill missing times between first and middle with first scan
                    for tm in range(all_times_sorted[0] + 1, t):
                        if tm in missing_times:
                            subject_times[tm] = subject_times[all_times_sorted[0]]

                    # fill missing times between middle and last with last scan
                    for tm in range(t + 1, all_times_sorted[-1]):
                        if tm in missing_times:
                            subject_times[tm] = subject_times[all_times_sorted[-1]]

                else:
                    # replicate missing scans based on nearest neighbor
                    for i in range(len(all_times_sorted) - 1):
                        t0, t1 = all_times_sorted[i], all_times_sorted[i + 1]
                        for tm in range(t0 + 1, t1):
                            if tm in missing_times:
                                if abs(tm - t0) <= abs(t1 - tm):
                                    subject_times[tm] = subject_times[t0]
                                else:
                                    subject_times[tm] = subject_times[t1] 

            full_time_data = {inverse_time_map[t]: arr for t, arr in subject_times.items()}
            full_time_data_sorted = sorted(full_time_data.items(), key=lambda x: global_time_map[x[0]])
            full_time_data_sorted_ = OrderedDict(full_time_data_sorted)

            # Save results
            save_path = os.path.join(save_folder, f"{subject_key}_interpolated.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(full_time_data_sorted_, f)

            print(f"Saved subject {subject_key} ({condi}) to {fold}")

    print("Done.")