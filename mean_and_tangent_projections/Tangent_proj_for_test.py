import os
import numpy as np
from geomstats.geometry.pre_shape import PreShapeSpace
import geomstats.backend as gs
from collections import defaultdict
import pickle

splits_root = r"C:\.....\ADNI\After_matching\Splitted"
folds = sorted([f for f in os.listdir(splits_root) if f.startswith("fold")])
for fold in folds:
    print("------ NOW RUNNING FOR ------{}".format(fold))
    condis_train = ['CN_train', 'AD_train']
    condis_test = ['CN_test', 'AD_test']

    m, k = 732, 3
    shape_space = PreShapeSpace(m, k)
    def frobenius_norm(matrix):
            return np.linalg.norm(matrix, ord='fro')
    
    for index, condi in enumerate(condis_test):
        save_folder_orig = r'C:\.....\ADNI\After_matching\Splitted\{}\{}_orig_shape_space'.format(fold, condi)
        os.makedirs(save_folder_orig, exist_ok=True)
        save_folder_tangent_data = r'C:\.....\ADNI\After_matching\Splitted\{}\{}_tangent_data'.format(fold, condi)
        os.makedirs(save_folder_tangent_data, exist_ok=True)

        folder_path = r'C:\.....\ADNI\After_matching\Splitted\{}\{}'.format(fold, condi)
        subject_timepoints = defaultdict(list)
        unique_timepoints = set()
        total_scans = 0

        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt') and filename.startswith('matched_'):
                total_scans += 1
                base = filename[len('matched_'):].replace('.txt', '')

                # Split subject ID and timepoint
                if '_' in base:
                    *subject_parts, timepoint = base.split('_')
                    subject_id = '_'.join(subject_parts)

                    subject_timepoints[subject_id].append(timepoint)
                    unique_timepoints.add(timepoint)

        # Count subjects without a 'bl' timepoint
        subjects_without_bl = [sid for sid, times in subject_timepoints.items() if 'bl' not in times]
        subjects_with_bl = [sid for sid, times in subject_timepoints.items() if 'bl' in times]

        print(f"Total number of subjects with 'bl': {len(subjects_with_bl)}\n")

        # we use the baseline of CN train as the common tangent to transport all vectors 
        with open(rf'C:\.....\ADNI\After_matching\Splitted\{fold}\{condis_train[0]}_Karcher_mean\{condis_train[0]}_Karcher_mean_shape.pkl', 'rb') as f:
            CN_temporal_means = pickle.load(f)
        baseline_mean = CN_temporal_means['bl']
        # take the corresponding dict mean
        with open(rf'C:\.....\ADNI\After_matching\Splitted\{fold}\{condis_train[0]}_Karcher_mean\{condis_train[0]}_Karcher_mean_shape.pkl', 'rb') as f:
            mean_shape_dict = pickle.load(f)

        # **** Projection ******
        def get_mean_shape_or_closest(timepoint, mean_shape_dict):
            if timepoint in mean_shape_dict:
                return mean_shape_dict[timepoint] 
            try:
                target_month = int(timepoint.replace('m', ''))
            except:
                raise ValueError(f"Unrecognized timepoint format: {timepoint}")
            available = []
            for k in mean_shape_dict.keys():
                if k.startswith('m'):
                    try:
                        val = int(k.replace('m', ''))
                        if val <= target_month:
                            available.append((val, k))
                    except:
                        continue
            if not available:
                raise ValueError(f"No valid fallback timepoint found for {timepoint}")        
            closest_key = max(available)[1]
            return mean_shape_dict[closest_key]

        for subject_id in subjects_with_bl:
            timepoints = subject_timepoints[subject_id]
            orig_shape_dict = {}
            tangent_dict = {}
            for timepoint in timepoints:
                filename = f"matched_{subject_id}_{timepoint}.txt"
                full_path = os.path.join(folder_path, filename)
                
                after_vectors = []
                # Read the data
                with open(full_path, 'r') as f:
                    for line in f:
                        before, after = line.split('->')
                        vector_after = np.array([float(x) for x in after.strip().split()])
                        after_vectors.append(vector_after)

                after_matrix = np.array(after_vectors)
                norm_after = frobenius_norm(after_matrix)
                normalized_after = after_matrix / norm_after
                mean_shape = get_mean_shape_or_closest(timepoint, mean_shape_dict)
                normalized_after_tan = shape_space.metric.log(normalized_after, mean_shape)

                if timepoint =='bl':
                    transported_vec = normalized_after_tan
                else:
                    # Parallel transport
                    transported_vec = shape_space.metric.parallel_transport(
                        normalized_after_tan,
                        mean_shape,
                        baseline_mean
                    )
                tangent_dict[timepoint] = transported_vec
                orig_shape_dict[timepoint] = normalized_after            
                            
            # save
            timepoint_str = '_'.join(sorted(str(k) for k in orig_shape_dict.keys()))
            save_filename = f"{subject_id}_{timepoint_str}.pkl"
            save_path_orig = os.path.join(save_folder_orig, save_filename)
            save_path_tan = os.path.join(save_folder_tangent_data, save_filename)
            # Save the orig and tangent_dict
            with open(save_path_orig, 'wb') as f:
                pickle.dump(orig_shape_dict, f)
            with open(save_path_tan, 'wb') as f:
                pickle.dump(tangent_dict, f)
        print("All completed")