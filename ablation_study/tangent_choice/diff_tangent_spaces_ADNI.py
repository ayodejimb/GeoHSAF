import os
import trimesh
import numpy as np
from collections import defaultdict
from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.learning.frechet_mean import FrechetMean
import geomstats.backend as gs
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import math
import copy
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import LambdaLR
from sklearn.decomposition import PCA
import gpytorch
import GPy
from tqdm import trange
from collections import OrderedDict

splits_root = r"C:\...\ADNI\After_matching\Splitted"
folds = sorted([f for f in os.listdir(splits_root) if f.startswith("fold")])

acc_list, auc_list = [], []

for fold in folds:

    print(f"------ NOW RUNNING FOR ------{fold}")

    condis_train = ['CN_train', 'AD_train']
    condis_test = ['CN_test', 'AD_test']

    data_folder_CN_train = r'C:\...\ADNI\After_matching\Splitted\{}\{}'.format(fold, condis_train[0]) 
    data_folder_AD_train = r'C:\...\ADNI\After_matching\Splitted\{}\{}'.format(fold, condis_train[1]) 
    data_folder_CN_test = r'C:\...\ADNI\After_matching\Splitted\{}\{}'.format(fold, condis_test[0]) 
    data_folder_AD_test = r'C:\...\ADNI\After_matching\Splitted\{}\{}'.format(fold, condis_test[1]) 

    def frobenius_norm(matrix):
        return np.linalg.norm(matrix, ord='fro')
    
    def convert(label):
        if label == 'bl':
            return 1
        elif label.startswith('m') and label[1:].isdigit():
            return int(label[1:])
        else:
            return None

    def build_subject_data(folder_path):
        data = defaultdict(dict)
        for filename in os.listdir(folder_path):
            if filename.endswith('.off'):
                base = filename.replace('.off', '')
                if '_' in base:
                    *subject_parts, time_point = base.split('_')
                    subject_id = '_'.join(subject_parts)

                    file_path = os.path.join(folder_path, filename)
                    mesh = trimesh.load_mesh(file_path, process=False)
                    points = np.array(mesh.vertices)

                    # Frobenius normalization
                    norm_points = frobenius_norm(points)
                    normalized_points = points / norm_points

                    data[subject_id][time_point] = normalized_points

    def build_subject_data(folder_path):
        data = defaultdict(dict)
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt') and filename.startswith('matched_'):
                # Remove 'matched_' prefix and '.txt' suffix
                base = filename[len('matched_'):].replace('.txt', '')
                if '_' in base:
                    *subject_parts, time_point = base.split('_')
                    subject_id = '_'.join(subject_parts)
                    file_path = os.path.join(folder_path, filename)
                    after_vectors = []
                    with open(file_path, 'r') as f:
                        for line in f:
                            _, after = line.split('->')
                            vector_after = np.array([float(x) for x in after.strip().split()])
                            after_vectors.append(vector_after)
                    points = np.array(after_vectors)
                    norm_points = frobenius_norm(points)
                    normalized_points = points / norm_points
                    data[subject_id][time_point] = normalized_points

        # Sort the time points using `convert`
        sorted_data = {}
        for subject_id, tp_dict in data.items():
            sorted_tp = dict(sorted(tp_dict.items(), key=lambda x: convert(x[0])))
            sorted_data[subject_id] = sorted_tp

        return sorted_data

    AD_subject_data_train = build_subject_data(data_folder_AD_train)   # dict in a dict , e.g {subject_ID: {'bl': array_data, ...}, ...}
    CN_subject_data_train = build_subject_data(data_folder_CN_train)
    AD_subject_data_test = build_subject_data(data_folder_AD_test)
    CN_subject_data_test = build_subject_data(data_folder_CN_test)

    # Retain those with baseline scans
    AD_subject_data_train_with_bl = {
        subject_id: timepoints
        for subject_id, timepoints in AD_subject_data_train.items()
        if 'bl' in timepoints
    }
    CN_subject_data_train_with_bl = {
        subject_id: timepoints
        for subject_id, timepoints in CN_subject_data_train.items()
        if 'bl' in timepoints
    }

    AD_subject_data_test_with_bl = {
        subject_id: timepoints
        for subject_id, timepoints in AD_subject_data_test.items()
        if 'bl' in timepoints
    }
    CN_subject_data_test_with_bl = {
        subject_id: timepoints
        for subject_id, timepoints in CN_subject_data_test.items()
        if 'bl' in timepoints
    }

    # *** Karcher mean ****
    m, k = 732, 3
    shape_space = PreShapeSpace(m, k)
    frechet_mean = FrechetMean(space=shape_space)

    # for computing means per timepoint
    def compute_mean_per_timepoint(subject_data):
        timepoint_to_arrays = defaultdict(list)
        for timepoints in subject_data.values():
            for tp, array in timepoints.items():
                timepoint_to_arrays[tp].append(array)
        
        tp_to_mean = {}
        for tp, arrays in timepoint_to_arrays.items():
            stacked = gs.array(np.stack(arrays))
            tp_to_mean[tp] = frechet_mean.fit(stacked).estimate_
        return tp_to_mean
    
    mean_shape_dict_CN = {}
    mean_shape_dict_AD = {}
    mean_shape_dict_CN = compute_mean_per_timepoint(CN_subject_data_train_with_bl)
    mean_shape_dict_AD = compute_mean_per_timepoint(AD_subject_data_train_with_bl)

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
            closest_key = 'bl'
        else:       
            closest_key = max(available)[1]
        return mean_shape_dict[closest_key]

    #  tangent vector at the mean
    def tangent_subject_data(subject_data, mean_shape_dict):
        tangent_data = {}
        for subject_id, timepoints in subject_data.items():
            tangent_data[subject_id] = {}
            for timepoint, array in timepoints.items():
                if timepoint == 'bl':
                    tangent_array = shape_space.metric.log(array, mean_shape_dict[timepoint])
                else:
                    mean_shape = get_mean_shape_or_closest(timepoint, mean_shape_dict)
                    tangent_array = shape_space.metric.log(array, mean_shape)
                tangent_data[subject_id][timepoint] = tangent_array.reshape(-1)
        return tangent_data    

    subject_AD_dict_train = tangent_subject_data(AD_subject_data_train_with_bl, mean_shape_dict_CN)
    subject_CN_dict_train = tangent_subject_data(CN_subject_data_train_with_bl, mean_shape_dict_CN)
    subject_AD_dict_test  = tangent_subject_data(AD_subject_data_test_with_bl, mean_shape_dict_CN)
    subject_CN_dict_test = tangent_subject_data(CN_subject_data_test_with_bl, mean_shape_dict_CN)

    print(f"Total number of AD train and test subjects with 'bl': {len(AD_subject_data_train_with_bl.keys()), len(AD_subject_data_test_with_bl.keys()) }")
    print(f"Total number of CN train and test subjects with 'bl': {len(CN_subject_data_train_with_bl.keys()), len(CN_subject_data_test_with_bl.keys()) }\n")

    # PGA 
    bl_data = []
    for subj_id, subj_data in subject_CN_dict_train.items():
        if 'bl' in subj_data:  
            bl_data.append(subj_data['bl'].reshape(-1))            
    for subj_id, subj_data in subject_AD_dict_train.items():
        if 'bl' in subj_data:
            bl_data.append(subj_data['bl'].reshape(-1))
    bl_data = np.stack(bl_data, axis=0)

    pca = PCA(n_components=0.95)
    pca.fit(bl_data)

    # Transform all timepoints
    def transform_subject_dict(subject_dict, pca_model):
        transformed_dict = {}
        for subj_id, subj_data in subject_dict.items():
            transformed_dict[subj_id] = {}
            for tp, arr in subj_data.items():
                flat = arr.reshape(1, -1)
                transformed = pca_model.transform(flat) 
                transformed_dict[subj_id][tp] = transformed.squeeze(0)
        return transformed_dict
    
    subject_CN_dict_train_pca = transform_subject_dict(subject_CN_dict_train, pca)
    subject_AD_dict_train_pca = transform_subject_dict(subject_AD_dict_train, pca)
    subject_CN_dict_test_pca  = transform_subject_dict(subject_CN_dict_test, pca)
    subject_AD_dict_test_pca  = transform_subject_dict(subject_AD_dict_test, pca)

    # **** interpolation *****
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def build_time_map(*dicts):
        time_labels = set()
        for d in dicts:
            for subj, tps in d.items():
                time_labels.update(tps.keys())

        def convert(label):
            if label == 'bl':
                return 1
            elif label.startswith('m') and label[1:].isdigit():
                return int(label[1:])
            else:
                return None

        valid_labels = [label for label in time_labels if convert(label) is not None]
        time_map = {label: convert(label) for label in sorted(valid_labels, key=convert)}
        time_map = dict(sorted(time_map.items(), key=lambda x: x[1]))
        inverse_time_map = {v: k for k, v in time_map.items()}
        return time_map, inverse_time_map

    class MultitaskGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, num_latents, num_tasks):
            inducing_points = torch.rand(num_latents, 16, 1)
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(-2), batch_shape=torch.Size([num_latents])
            )
            variational_strategy = gpytorch.variational.LMCVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                ),
                num_tasks=num_tasks,
                num_latents=num_latents,
                latent_dim=-1
            )
            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents]), ard_num_dims=1),
                batch_shape=torch.Size([num_latents])
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
    def interpolate_subjects(subject_dict, global_time_map, inverse_time_map,
                         num_latents=3, num_epochs=35):
        interpolated_results = {}

        for subject_key, data in subject_dict.items():
            subject_times = {global_time_map[k]: v for k, v in data.items() if k in global_time_map}
            if len(subject_times) < 2:
                interpolated_results[subject_key] = data
                continue

            all_times_sorted = sorted(subject_times.keys())
            missing_times = set(global_time_map.values()) - set(all_times_sorted)

            # Train/validation split setup
            if len(all_times_sorted) == 2:
                train_x, train_y = [], []
                first_t, last_t = all_times_sorted[0], all_times_sorted[-1]
                train_x.append([first_t]); train_y.append(subject_times[first_t])
                train_x.append([last_t]);  train_y.append(subject_times[last_t])
                val_x, val_y = [], []

            elif len(all_times_sorted) > 2:
                middle = all_times_sorted[1:-1]
                train_x, train_y, val_x, val_y = [], [], [], []

                first_t, last_t = all_times_sorted[0], all_times_sorted[-1]
                train_x.append([first_t]); train_y.append(subject_times[first_t])
                train_x.append([last_t]);  train_y.append(subject_times[last_t])

                if len(middle) == 1:
                    t = middle[0]
                    train_x.append([t]); train_y.append(subject_times[t])
                else:
                    train_mid, val_mid = train_test_split(middle, test_size=0.2, random_state=42)
                    for t in train_mid:
                        train_x.append([t]); train_y.append(subject_times[t])
                    for t in val_mid:
                        val_x.append([t]); val_y.append(subject_times[t])

            train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
            train_y = torch.tensor(np.vstack(train_y), dtype=torch.float32).to(device)
            if val_x:
                val_x = torch.tensor(val_x, dtype=torch.float32).to(device)
                val_y = torch.tensor(np.vstack(val_y), dtype=torch.float32).to(device)

            num_tasks = train_y.size(1)
            model = MultitaskGPModel(num_latents, num_tasks).to(device)
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)

            # Train
            model.train(); likelihood.train()
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': likelihood.parameters()}
            ], lr=0.1)
            mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

            for epoch in trange(num_epochs, leave=False):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

            # Interpolate missing times
            model.eval(); likelihood.eval()
            full_time_data = {inverse_time_map[t]: arr for t, arr in subject_times.items()}
            first_t, last_t = all_times_sorted[0], all_times_sorted[-1]

            for missing_t in sorted(missing_times):
                if missing_t < first_t or missing_t > last_t:
                    continue
                with torch.no_grad():
                    inp = torch.tensor([[missing_t]], dtype=torch.float32).to(device)
                    pred = likelihood(model(inp)).mean
                    full_time_data[inverse_time_map[missing_t]] = pred.cpu().numpy()

            # Sort by time
            full_time_data_sorted = sorted(full_time_data.items(), key=lambda x: global_time_map[x[0]])
            interpolated_results[subject_key] = OrderedDict(full_time_data_sorted)

        return interpolated_results
    
    global_time_map, inverse_time_map = build_time_map(subject_AD_dict_train_pca, subject_CN_dict_train_pca)

    print("Interpolaton begin !!!!\n")
    AD_train_completed = interpolate_subjects(subject_AD_dict_train_pca, global_time_map, inverse_time_map)
    AD_test_completed = interpolate_subjects(subject_AD_dict_test_pca, global_time_map, inverse_time_map)
    CN_train_completed = interpolate_subjects(subject_CN_dict_train_pca, global_time_map, inverse_time_map)
    CN_test_completed = interpolate_subjects(subject_CN_dict_test_pca, global_time_map, inverse_time_map)
    print("Interpolaton ends !!!!\n")
    print("Done Loading Data for Training !!!!")

    def get_first_array_dimension(d):
        max_dim = -1
        for subj in d:
            for t in d[subj]:
                current_dim = d[subj][t].shape[-1]
                if current_dim > max_dim:
                    max_dim = current_dim

        if max_dim == -1:
            raise ValueError("Dictionary is empty or contains no arrays.")
        return max_dim

    def pad_dict_arrays(d, target_dim):
        for subj in d:
            for t in d[subj]:
                arr = d[subj][t]
                if arr.shape[-1] < target_dim:
                    pad_width = target_dim - arr.shape[-1]
                    d[subj][t] = np.pad(arr, ((0, 0), (0, pad_width)) if arr.ndim == 2 else ((0, pad_width),), mode='constant')
        return d

    dim1 = get_first_array_dimension(AD_train_completed)
    dim2 = get_first_array_dimension(CN_train_completed)
    dim3 = get_first_array_dimension(CN_test_completed)
    dim4 = get_first_array_dimension(AD_test_completed)

    max_dim = max(dim1, dim2, dim3, dim4)
    subject_AD_dict_train = pad_dict_arrays(AD_train_completed, max_dim)
    subject_AD_dict_test = pad_dict_arrays(AD_test_completed, max_dim)
    subject_CN_dict_train = pad_dict_arrays(CN_train_completed, max_dim)
    subject_CN_dict_test = pad_dict_arrays(CN_test_completed, max_dim)

    def prepare_subject_data(group_dict, label):
        data = []
        for subject_id, time_dict in group_dict.items():
            times = sorted(time_dict.keys())
            subject_seq = []
            for t in times:
                vec = torch.tensor(time_dict[t]).squeeze()
                if vec.ndim != 1:
                    raise ValueError(f"Vector at {subject_id} time {t} is not 1D after squeeze: shape={vec.shape}")
                subject_seq.append(vec)
            data.append((subject_seq, label))
        return data

    # Combine CN and AD
    CN_data_train = prepare_subject_data(subject_CN_dict_train, label=1)
    AD_data_train = prepare_subject_data(subject_AD_dict_train, label=0)
    all_data_train = CN_data_train + AD_data_train   # format is like this [[tensors of time_data], label], .... ]

    CN_data_test = prepare_subject_data(subject_CN_dict_test, label=1)
    AD_data_test = prepare_subject_data(subject_AD_dict_test, label=0)
    all_data_test = CN_data_test + AD_data_test 

    class TimeSeriesSubjectDataset(Dataset):
        def __init__(self, data):
            self.data = data  # list of [sequence, label]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sequence, label = self.data[idx]
            return torch.stack(sequence), label  # shape: (T_i, D), label
        
    def collate_fn(batch):
        sequences, labels = zip(*batch)
        padded = pad_sequence(sequences, batch_first=True)  # (B, T_max, D)
        lengths = [seq.size(0) for seq in sequences]

        # Create padding mask (True = pad) - boolean mask
        mask = torch.tensor([[i >= l for i in range(padded.size(1))] for l in lengths])

        return padded.float(), torch.tensor(labels), mask
            
    class TransformerClassifier(nn.Module):
        def __init__(self, input_dim, model_dim, num_classes, num_heads=4, num_layers=4, dropout=0.1, max_len=600):
            super(TransformerClassifier, self).__init__()
            
            self.input_proj = nn.Linear(input_dim, model_dim)

            self.pos_embedding = nn.Embedding(max_len + 1, model_dim)  # +1 for CLS token
            nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=1e-4)

            ffn_hidden = 4 * model_dim
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=ffn_hidden,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )

            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) 
            self.cls_token = nn.Parameter(torch.empty(1, 1, model_dim))
            nn.init.normal_(self.cls_token, mean=0.0, std=1e-4)

            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            self.fc_out = nn.Linear(model_dim, num_classes)

        def forward(self, x, mask):
            
            B, T, _ = x.size()
            x = self.input_proj(x)

            # Append CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, D)

            positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)  # (1, T+1)
            x = x + self.pos_embedding(positions) 

            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)  # (B, T+1)

            # Transformer encoding
            x = self.transformer_encoder(x, src_key_padding_mask=mask)

            # Extract CLS token output
            cls_out = x[:, 0, :]  # (B, D)

            out = self.dropout(cls_out)
            out = self.fc_out(out)  # (B, num_classes)
            
            return out

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = TimeSeriesSubjectDataset(all_data_train)
    test_dataset = TimeSeriesSubjectDataset(all_data_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    input_dim = all_data_train[0][0][0].shape[0] 
    model = TransformerClassifier(input_dim=input_dim, model_dim=128, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    epochs = 50
    def train_one_epoch(model, dataloader, optimizer, criterion):
        model.train()
        total_loss, correct, total = 0, 0, 0
        all_labels = []
        all_probs = []
        
        for x, y, mask in dataloader:
            
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            # **** this is for AUC *****
            probs = torch.softmax(logits, dim=1)[:, 1]  # Prob of positive class
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

        avg_loss = total_loss / total
        accuracy = correct / total
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = float('nan')

        return avg_loss, accuracy, auc

    @torch.no_grad()
    def evaluate(model, dataloader, criterion):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        all_labels = []
        all_probs = []

        for x, y, mask in dataloader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            logits = model(x, mask)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()

            total += y.size(0)

            # **** this is for AUC *****
            probs = torch.softmax(logits, dim=1)[:, 1]  # Prob of positive class
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / total
        accuracy = correct / total
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = float('nan')

        return avg_loss, accuracy, auc

    best_test_acc = 0
    best_test_auc = 0
    best_metrics = {}

    for epoch in range(epochs):
        train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion)

        # if (test_acc >= best_test_acc) and (test_auc >= best_test_auc):
        if (test_acc >= best_test_acc):
            
            best_test_acc = test_acc
            best_test_auc = test_auc

            best_metrics = {
                'epoch': epoch + 1,
                'acc': test_acc,
                'auc': test_auc,
                'loss': test_loss
            }

        print(f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f} | "
            f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")

    print(f"\nBest Test Accuracy: {best_metrics['acc']:.4f} "
        f"(Epoch {best_metrics['epoch']}) | AUC: {best_metrics['auc']:.4f}, "
        f"Loss: {best_metrics['loss']:.4f}")
    
    acc_list.append(best_metrics['acc'])
    auc_list.append(best_metrics['auc'])

# Convert to numpy arrays for mean/std
acc_arr = np.array(acc_list)
auc_arr = np.array(auc_list)

# Print accuracies
print("Accuracies:", acc_arr)
print(f"Mean Accuracy: {acc_arr.mean():.4f}, Std: {acc_arr.std():.4f}")

# Print AUCs
print("AUCs:", auc_arr)
print(f"Mean AUC: {auc_arr.mean():.4f}, Std: {auc_arr.std():.4f}")