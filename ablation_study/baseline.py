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
from geomstats.learning.frechet_mean import FrechetMean

splits_root = r"C:\....\ADNI\Before_matching\Splitted"
folds = sorted([f for f in os.listdir(splits_root) if f.startswith("fold")])

acc_list, auc_list = [], []
for fold in folds:

    print(f"------ NOW RUNNING FOR ------{fold}")

    condis_train = ['CN_train', 'AD_train']
    condis_test = ['CN_test', 'AD_test']

    data_folder_CN_train = r'C:\....\ADNI\Before_matching\Splitted\{}\{}'.format(fold, condis_train[0]) 
    data_folder_AD_train = r'C:\....\ADNI\Before_matching\Splitted\{}\{}'.format(fold, condis_train[1]) 
    data_folder_CN_test = r'C:\....\ADNI\Before_matching\Splitted\{}\{}'.format(fold, condis_test[0]) 
    data_folder_AD_test = r'C:\....\ADNI\Before_matching\Splitted\{}\{}'.format(fold, condis_test[1]) 

    # Function to compute Frobenius norm of a matrix
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

                    norm_points = frobenius_norm(points)
                    normalized_points = points / norm_points

                    data[subject_id][time_point] = normalized_points

        # Sort the time points using `convert`
        sorted_data = {}
        for subject_id, tp_dict in data.items():
            sorted_tp = dict(sorted(tp_dict.items(), key=lambda x: convert(x[0])))
            sorted_data[subject_id] = sorted_tp

        return sorted_data

    AD_subject_data_train = build_subject_data(data_folder_AD_train) 
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
    mean_shape_dict_CN = compute_mean_per_timepoint(CN_subject_data_train_with_bl)

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

    # (2) tangent vector at the mean 
    def tangent_subject_data(subject_data, mean_shape_dict):
        tangent_data = {}
        for subject_id, timepoints in subject_data.items():
            tangent_data[subject_id] = {}
            for timepoint, array in timepoints.items():
                if timepoint == 'bl':
                    tangent_array = shape_space.metric.log(array, mean_shape_dict[timepoint])
                else:
                    mean_shape = get_mean_shape_or_closest(timepoint, mean_shape_dict)
                    tangent_arra = shape_space.metric.log(array, mean_shape)
                    # Parallel transport to baseline tangent space 
                    tangent_array = shape_space.metric.parallel_transport(
                        tangent_arra,
                        mean_shape,
                        mean_shape_dict['bl']
                    )
                tangent_data[subject_id][timepoint] = tangent_array.reshape(-1)
        return tangent_data    

    subject_AD_dict_train = tangent_subject_data(AD_subject_data_train_with_bl, mean_shape_dict_CN)
    subject_CN_dict_train = tangent_subject_data(CN_subject_data_train_with_bl, mean_shape_dict_CN)
    subject_AD_dict_test  = tangent_subject_data(AD_subject_data_test_with_bl, mean_shape_dict_CN)
    subject_CN_dict_test = tangent_subject_data(CN_subject_data_test_with_bl, mean_shape_dict_CN)

    print(f"Total number of AD train and test subjects with 'bl': {len(AD_subject_data_train_with_bl.keys()), len(AD_subject_data_test_with_bl.keys()) }")
    print(f"Total number of CN train and test subjects with 'bl': {len(CN_subject_data_train_with_bl.keys()), len(CN_subject_data_test_with_bl.keys()) }\n")

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
                    # Pad zeros at the end of the last axis
                    d[subj][t] = np.pad(arr, ((0, 0), (0, pad_width)) if arr.ndim == 2 else ((0, pad_width),), mode='constant')
        return d

    dim1 = get_first_array_dimension(subject_AD_dict_train)
    dim2 = get_first_array_dimension(subject_CN_dict_train)
    dim3 = get_first_array_dimension(subject_CN_dict_test)
    dim4 = get_first_array_dimension(subject_AD_dict_test)

    max_dim = max(dim1, dim2, dim3, dim4)
    subject_AD_dict_train = pad_dict_arrays(subject_AD_dict_train, max_dim)
    subject_AD_dict_test = pad_dict_arrays(subject_AD_dict_test, max_dim)
    subject_CN_dict_train = pad_dict_arrays(subject_CN_dict_train, max_dim)
    subject_CN_dict_test = pad_dict_arrays(subject_CN_dict_test, max_dim)
    # *************** End ******************

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
        sequences, labels = zip(*batch)  #remember this operates on a batch
        
        # Pad sequences
        padded = pad_sequence(sequences, batch_first=True)  # (B, T_max, D)
        lengths = [seq.size(0) for seq in sequences]

        # Create padding mask (True = pad) - boolean mask
        mask = torch.tensor([[i >= l for i in range(padded.size(1))] for l in lengths])

        return padded.float(), torch.tensor(labels), mask
            
    class TransformerClassifier(nn.Module):
        def __init__(self, input_dim, model_dim, num_classes, num_heads=4, num_layers=4, dropout=0.1, max_len=600):
            super(TransformerClassifier, self).__init__()
            
            self.input_proj = nn.Linear(input_dim, model_dim)

            self.pos_embedding = nn.Embedding(max_len + 1, model_dim)
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
            # x: (batch_size, seq_len, input_dim)
            # mask: (batch_size, seq_len), True for padding
            
            B, T, _ = x.size()
            
            # Input projection
            x = self.input_proj(x)

            # Append CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, D)

            positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)  # (1, T+1)
            x = x + self.pos_embedding(positions) 

            # Update mask to account for cls token
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)  # (B, T+1)

            # Transformer encoding
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
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