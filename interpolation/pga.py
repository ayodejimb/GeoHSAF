import os
import pickle
import numpy as np
from glob import glob
from geomstats.geometry.pre_shape import PreShapeSpace
from sklearn.decomposition import PCA
import random
import logging
from pathlib import Path
from matplotlib.pyplot import plt
random.seed(42)
np.random.seed(42)

splits_root = r"C:\.....\ADNI\After_matching\Splitted"
folds = sorted([f for f in os.listdir(splits_root) if f.startswith("fold")])

# components/direction lists
n_components_list = [round(x, 2) for x in np.arange(0.1, 1.0, 0.05)]
m, k = 732, 3
shape_space = PreShapeSpace(m, k)

def load_pickle(p):
    with open(p, 'rb') as f:
        return pickle.load(f)
    
def save_pickle(obj, p):
    with open(p, 'wb') as f:
        pickle.dump(obj, f)

def subject_id_from_filename(fname):
    base = os.path.basename(fname)
    if '_bl' in base:
        return base.split('_bl')[0]
    return os.path.splitext(base)[0]

def flatten_tangent(t):
    return np.array(t).reshape(-1)

def unflatten_tangent(flat):
    flat = np.asarray(flat).ravel()
    expected = m * k
    if flat.size != expected:
        raise ValueError(f"Cannot unflatten tangent: expected {expected} elements (m*k), got {flat.size}")
    return flat.reshape(m, k)

def process_reconstructed_value(value, mean_shape):
    value = np.asarray(value)
    if value.shape != (m, k):
        raise ValueError(f"process_reconstructed_value: expected tangent shape {(m,k)}, got {value.shape}")
    return shape_space.metric.exp(value, mean_shape)

def compute_mae(reconstructed_data, true_data):
    return np.mean(np.abs(reconstructed_data - true_data))

def list_pkl(folder):
    p = Path(folder)
    if not p.exists():
        return []
    return sorted([str(x) for x in p.glob('*.pkl')])

# load bl tangents from tangent folder: returns dict subject_id -> { 'filename':filename, 'bl':array, 'full':loaded_dict }
def load_bl_tangents_from_folder(tangent_folder):
    items = {}
    for p in list_pkl(tangent_folder):
        data = load_pickle(p)
        if 'bl' not in data:
            print(f"Warning: Baseline ('bl') not available in {p}. Check for error but skipping.")
            continue
        subj = subject_id_from_filename(os.path.basename(p))
        items[subj] = {'filename': p, 'bl': np.array(data['bl']), 'full': data}
    return items

# load original (shape space) data for subjects: returns dict subject_id -> full dict
def load_original_folder(orig_folder):
    items = {}
    for p in list_pkl(orig_folder):
        data = load_pickle(p)
        subj = subject_id_from_filename(os.path.basename(p))
        items[subj] = {'filename': p, 'full': data}
    return items

# balance two dicts by downsampling the larger to match the smaller
def balance_keys(dictA, dictB):
    keysA = list(dictA.keys())
    keysB = list(dictB.keys())
    nA, nB = len(keysA), len(keysB)
    if nA == 0 or nB == 0:
        raise ValueError('One of the shapes dicitonary is empty')
    if nA > nB:
        chosenA = random.sample(keysA, nB)
        chosenB = keysB
    elif nB > nA:
        chosenB = random.sample(keysB, nA)
        chosenA = keysA
    else:
        chosenA, chosenB = keysA, keysB
    return chosenA, chosenB

# -----------------------------------------------------------------
# Main loop across folds
# -----------------------------------------------------------------
for fold in folds:
    print(f"------ NOW RUNNING FOR ------{fold}")
    fold_path = os.path.join(splits_root, fold)

    # load Karcher mean reference from CN_train_Karcher_mean
    karcher_folder = os.path.join(fold_path, 'CN_train_Karcher_mean')
    mean_shape = None
    if os.path.isdir(karcher_folder):
        files = list_pkl(karcher_folder)
        if len(files) > 0:
            mean_data = load_pickle(files[0])
            mean_shape = np.array(mean_data['bl']) if isinstance(mean_data, dict) and 'bl' in mean_data else print('Mean Cannot be found')
    if mean_shape is None:
        raise RuntimeError(f"Could not load Karcher mean for fold {fold} from {karcher_folder}")

    # folders for CN/AD train and test (tangent and original)
    CN_train_tangent = os.path.join(fold_path, 'CN_train_tangent_data')
    AD_train_tangent = os.path.join(fold_path, 'AD_train_tangent_data')
    CN_test_tangent = os.path.join(fold_path, 'CN_test_tangent_data')
    AD_test_tangent = os.path.join(fold_path, 'AD_test_tangent_data')

    CN_train_orig = os.path.join(fold_path, 'CN_train_orig_shape_space')
    AD_train_orig = os.path.join(fold_path, 'AD_train_orig_shape_space')
    CN_test_orig = os.path.join(fold_path, 'CN_test_orig_shape_space')
    AD_test_orig = os.path.join(fold_path, 'AD_test_orig_shape_space')

    # load bl tangents
    CN_train_dict = load_bl_tangents_from_folder(CN_train_tangent)
    AD_train_dict = load_bl_tangents_from_folder(AD_train_tangent)
    CN_test_dict = load_bl_tangents_from_folder(CN_test_tangent)
    AD_test_dict = load_bl_tangents_from_folder(AD_test_tangent)

    # load originals for ground truth
    CN_train_orig_dict = load_original_folder(CN_train_orig)
    AD_train_orig_dict = load_original_folder(AD_train_orig)
    CN_test_orig_dict = load_original_folder(CN_test_orig)
    AD_test_orig_dict = load_original_folder(AD_test_orig)

    # to balance training sets
    chosen_CN, chosen_AD = balance_keys(CN_train_dict, AD_train_dict)
    CN_train_bal = {k: CN_train_dict[k] for k in chosen_CN}
    AD_train_bal = {k: AD_train_dict[k] for k in chosen_AD}

    # build training matrix (flattened tangents)
    X_CN = np.stack([flatten_tangent(v['bl']) for v in CN_train_bal.values()], axis=0)
    X_AD = np.stack([flatten_tangent(v['bl']) for v in AD_train_bal.values()], axis=0)
    X_train = np.vstack([X_CN, X_AD])

    pca_models = []
    explained_variances = []
    mse_train_list = []
    mse_CN_train_list = []
    mse_AD_train_list = []
    mse_CN_test_list = []
    mse_AD_test_list = []

    # iterate PGA candidates (n_components as explained variance fraction)
    for n_comp in n_components_list:
        print(f"Fitting PCA with n_components={n_comp}")
        pca = PCA(n_components=n_comp)
        pca.fit(X_train)   # PGA fitted on train samples

        # evaluate on training samples
        def eval_dict_on_pca(pca_obj, data_dict, orig_dict):
            keys = list(data_dict.keys())
            if len(keys) == 0:
                return []
            X = np.stack([flatten_tangent(data_dict[k]['bl']) for k in keys], axis=0)
            X_proj = pca_obj.transform(X)
            X_rec = pca_obj.inverse_transform(X_proj)
            maes = []
            for i, k in enumerate(keys):
                rec_tangent = unflatten_tangent(X_rec[i])
                rec_shape = process_reconstructed_value(rec_tangent, mean_shape)
                # get true shape from orig_dict (use subject id)
                subj = k
                if subj in orig_dict:
                    true_shape = np.array(orig_dict[subj]['full']['bl']) if 'bl' in orig_dict[subj]['full'] else np.array(orig_dict[subj]['full'])
                else:
                    # fallback to using tangent
                    true_shape = data_dict[k]['bl']
                maes.append(compute_mae(rec_shape, true_shape))
            return maes

        maes_CN_train = eval_dict_on_pca(pca, CN_train_bal, CN_train_orig_dict)
        maes_AD_train = eval_dict_on_pca(pca, AD_train_bal, AD_train_orig_dict)
        maes_CN_test = eval_dict_on_pca(pca, CN_test_dict, CN_test_orig_dict)
        maes_AD_test = eval_dict_on_pca(pca, AD_test_dict, AD_test_orig_dict)

        # aggregate
        pca_models.append(pca)
        explained_variances.append(np.sum(pca.explained_variance_ratio_))
        mse_CN_train_list.append(np.nanmean(maes_CN_train) if len(maes_CN_train)>0 else np.nan)
        mse_AD_train_list.append(np.nanmean(maes_AD_train) if len(maes_AD_train)>0 else np.nan)
        mse_CN_test_list.append(np.nanmean(maes_CN_test) if len(maes_CN_test)>0 else np.nan)
        mse_AD_test_list.append(np.nanmean(maes_AD_test) if len(maes_AD_test)>0 else np.nan)
        mse_train_list.append(np.nanmean(maes_CN_train + maes_AD_train) if (len(maes_CN_train)+len(maes_AD_train))>0 else np.nan)

        print(f"Explained var: {explained_variances[-1]:.4f} | Train MAE (CN/AD): {mse_CN_train_list[-1]:.4f}/{mse_AD_train_list[-1]:.4f} | Test MAE (CN/AD): {mse_CN_test_list[-1]:.4f}/{mse_AD_test_list[-1]:.4f}")

    # choose best PGA by lowest mean train error across CN and AD (average)
    mean_train_errors = [np.nanmean([a, b]) for a, b in zip(mse_CN_train_list, mse_AD_train_list)]
    best_idx = int(np.nanargmin(mean_train_errors))
    best_pca = pca_models[best_idx]

    # save PGA models and summary
    save_pca_folder = os.path.join(fold_path, 'PCA_model')
    os.makedirs(save_pca_folder, exist_ok=True)
    save_pickle(best_pca, os.path.join(save_pca_folder, f'{fold}_pca.pkl'))
    
    summary = {
        'n_components_list': n_components_list,
        'explained_variances': explained_variances,
        'mse_CN_train_list': mse_CN_train_list,
        'mse_AD_train_list': mse_AD_train_list,
        'mse_CN_test_list': mse_CN_test_list,
        'mse_AD_test_list': mse_AD_test_list,
        'best_idx': best_idx,
    }
    save_pickle(summary, os.path.join(save_pca_folder, 'summary_stats.pkl'))

    # plot error vs explained variance
    plt.figure(figsize=(8,5))
    plt.plot(explained_variances, [np.nanmean([a,b]) for a,b in zip(mse_CN_train_list, mse_AD_train_list)], marker='o', label='Train Mean MAE')
    plt.plot(explained_variances, [np.nanmean([a,b]) for a,b in zip(mse_CN_test_list, mse_AD_test_list)], marker='s', label='Test Mean MAE')
    plt.xlabel('PCA Explained Variance')
    plt.ylabel('Mean Reconstruction MAE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(fold_path, 'pca_vs_error.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Best PCA at index {best_idx} (n_comp={n_components_list[best_idx]}), saved to {save_pca_folder}")

    # Project all timepoints for each subject in CN_train, AD_train, CN_test, AD_test using best_pca and save
    def project_and_save_all(input_tangent_dict, out_folder):
        os.makedirs(out_folder, exist_ok=True)
        for subj, info in input_tangent_dict.items():
            filename = os.path.basename(info['filename'])
            full_dict = info['full']
            new_dict = {}
            for tp, arr in full_dict.items():
                flat = flatten_tangent(arr)
                proj = best_pca.transform(flat.reshape(1, -1))
                new_dict[tp] = proj
            base_name = os.path.splitext(filename)[0]
            save_name = f"{base_name}_pca_result.pkl"
            save_path = os.path.join(out_folder, save_name)
            save_pickle(new_dict, save_path)

    CN_train_out = os.path.join(fold_path, 'CN_train_PCA_tangent_data')
    AD_train_out = os.path.join(fold_path, 'AD_train_PCA_tangent_data')
    CN_test_out = os.path.join(fold_path, 'CN_test_PCA_tangent_data')
    AD_test_out = os.path.join(fold_path, 'AD_test_PCA_tangent_data')

    project_and_save_all(CN_train_dict, CN_train_out)
    project_and_save_all(AD_train_dict, AD_train_out)
    project_and_save_all(CN_test_dict, CN_test_out)
    project_and_save_all(AD_test_dict, AD_test_out)

    print(f"Finished for {fold}. Results and plots saved in {fold_path}.")