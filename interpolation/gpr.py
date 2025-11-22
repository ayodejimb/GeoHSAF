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

splits_root = r"C:\.....\ADNI\After_matching\Splitted"
folds = sorted([f for f in os.listdir(splits_root) if f.startswith("fold")])

for fold in folds:
    print(f"------ NOW RUNNING FOR ------{fold}")

    base_path = r"C:\.....\ADNI\After_matching\Splitted\{}".format(fold)
    # Extract GLOBAL time map
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

    # Define Multitask GP
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

    # Per-subject training + interpolation
    condis = ['CN_train', 'AD_train', 'CN_test', 'AD_test']
    num_latents = 3
    for condi in condis:
        data_folder = os.path.join(base_path, f"{condi}_PCA_tangent_data")
        save_folder = os.path.join(base_path, f"{condi}_interpolated_PCA_tangent_data")
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

                # we save, nothing to interpolate ('interpolated' added only to the name for consistency)
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

                    # we save, nothing to interpolate ('interpolated' added only to the name for consistency)
                    save_path = os.path.join(save_folder, f"{subject_key}_interpolated.pkl")
                    with open(save_path, 'wb') as f:
                        pickle.dump(data, f)
                    print(f"Saved subject {subject_key} ({condi}) to {fold}")

                    continue

                else:
                    # Build training here (first and last), no validation because we have 2 scans and need to interpolate between
                    train_x, train_y = [], []
                    val_x, val_y = [], []

                    # include first and last as train
                    first_t, last_t = all_times_sorted[0], all_times_sorted[-1]
                    train_x.append([first_t]); train_y.append(subject_times[first_t])
                    train_x.append([last_t]);  train_y.append(subject_times[last_t])

            if len(all_times_sorted) > 2:
                middle = all_times_sorted[1:-1]

                train_x, train_y = [], []
                val_x, val_y = [], []

                # include first and last as train
                first_t, last_t = all_times_sorted[0], all_times_sorted[-1]
                train_x.append([1, first_t]); train_y.append(subject_times[first_t])
                train_x.append([1, last_t]);  train_y.append(subject_times[last_t])

                if len(middle) == 1:
                    # Only one middle scan: add to train
                    t = middle[0]
                    train_x.append([t]); train_y.append(subject_times[t])

                else:
                    # Random split of middle scans into train and val
                    train_mid, val_mid = train_test_split(
                        middle, test_size=0.2, random_state=42
                    )
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
            num_epochs = 35

            for epoch in trange(num_epochs, leave=False):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

            # Interpolate missing scans
            model.eval(); likelihood.eval()
            full_time_data = {inverse_time_map[t]: arr for t, arr in subject_times.items()}
            for missing_t in sorted(missing_times):
                if missing_t < first_t or missing_t > last_t:

                    continue  # only interpolate within trajectory range
                    
                with torch.no_grad():
                    inp = torch.tensor([[missing_t]], dtype=torch.float32).to(device)
                    pred = likelihood(model(inp)).mean
                    full_time_data[inverse_time_map[missing_t]] = pred.cpu().numpy()

            full_time_data_sorted = sorted(full_time_data.items(), key=lambda x: global_time_map[x[0]])
            full_time_data_sorted_ = OrderedDict(full_time_data_sorted)

            # Save results
            save_path = os.path.join(save_folder, f"{subject_key}_interpolated.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(full_time_data_sorted_, f)

            print(f"Saved subject {subject_key} ({condi}) to {fold}")

    print("Done.")