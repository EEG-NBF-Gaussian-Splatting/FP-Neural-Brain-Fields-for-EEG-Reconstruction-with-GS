import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import configargparse
from Run_EEG_GS_Helpers import GaussianEEGField
from torch.utils.data import Dataset, DataLoader
import time
import mne
from copy import deepcopy

torch.set_default_tensor_type('torch.cuda.FloatTensor') 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def denormalize_voltage(normalized, norm_volt_params):


    #Reverse the z-score normalization
    

    mean_val, std_val = norm_volt_params
    

    denormalized = normalized * std_val + mean_val

    return denormalized




def normalize_voltage(voltage, norm_params=None):


    if norm_params is not None:
        mean_val, std_val = norm_params
    else:
        mean_val = torch.mean(voltage)
        std_val = torch.std(voltage)
    
    # Handle the case where std is 0
    if std_val == 0:
        return torch.zeros_like(voltage), (mean_val, std_val)
    
    # Z-score normalization
    normalized = (voltage - mean_val) / std_val

    return normalized, (mean_val, std_val)
class VoltageDataset(Dataset):
    def __init__(self, points, voltages, train_voltage_norm_params=None):
        self.points = points
        self.voltages = voltages
        
        # If train_voltage_norm_params is provided, use those; otherwise calculate new ones
        if train_voltage_norm_params is not None:
            self.normalized_voltages, _ = normalize_voltage(self.voltages, train_voltage_norm_params)
            self.voltage_norm_params = train_voltage_norm_params
        else:
            self.normalized_voltages, self.voltage_norm_params = normalize_voltage(self.voltages)
        
        assert len(points) == len(voltages), "Points and voltages must have same length"
        self.points = self.points.to(dtype=torch.float32)
        self.voltages = self.voltages.to(dtype=torch.float32)
        self.normalized_voltages = self.normalized_voltages.to(dtype=torch.float32)
    
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, idx):
        return self.points[idx], self.voltages[idx], self.normalized_voltages[idx], self.voltage_norm_params


def get_coordinate_normalization_params(vector):
    coordinates = vector[:, :3]
    time = vector[:, 3]
    
    coord_min = torch.min(coordinates, dim=0)[0]
    coord_max = torch.max(coordinates, dim=0)[0]
    time_min = torch.min(time)
    time_max = torch.max(time)
    
    return {
        'coord_min': coord_min,
        'coord_max': coord_max,
        'time_min': time_min,
        'time_max': time_max
    }


def normalize_coordinates(vector, train_norm_params):
    coordinates = vector[:, :3]
    time = vector[:, 3]
    
    normalized_coordinates = 2 * (coordinates - train_norm_params['coord_min']) / \
                           (train_norm_params['coord_max'] - train_norm_params['coord_min']) - 1
    
    normalized_time = (time - train_norm_params['time_min']) / \
                     (train_norm_params['time_max'] - train_norm_params['time_min'])
    
    return torch.cat([normalized_coordinates, normalized_time.unsqueeze(1)], dim=1)

def add_auditory_electrodes(raw, skip_numbers=None):

    from mne.channels import make_standard_montage
    
    print("Generating auditory-specific electrode positions ")
    
    if skip_numbers is None:
        skip_numbers = []
    
    # skip 29, original recording does not have electrode number 29
    if 29 not in skip_numbers:
        skip_numbers.append(29)
    
    existing_ch_names = raw.ch_names
    
    standard_montage = make_standard_montage('standard_1005')
    std_ch_positions = standard_montage.get_positions()['ch_pos']
    
    prioritized_auditory = [
       
        'T7', 'T8',           # Mid-temporal
        'TP7', 'TP8',         # Temporal-parietal junction
        'FC5', 'FC6',         # Fronto-central 
        'FT7', 'FT8',         # Fronto-temporal
        'P7', 'P8',           # Posterior temporal
        'CP5', 'CP6',         # Centro-parietal
        'C5', 'C6',           # Central lateral areas
        'F7', 'F8',           # Lateral frontal
        'P5', 'P6'            
    ]
    
    # Map old names to new names for duplicate checking
    old_to_new = {'T3':'T7', 'T4':'T8', 'T5':'P7', 'T6':'P8'}
    new_to_old = {v:k for k,v in old_to_new.items()}
    
    available_positions = []
    for ch in prioritized_auditory:
        if ch in existing_ch_names:
            print(f"Skipping {ch} - already exists in recording")
            continue
            
        old_name = new_to_old.get(ch)
        if old_name and old_name in existing_ch_names:
            print(f"Skipping {ch} - equivalent position {old_name} exists")
            continue
        
        should_skip = False
        for num in skip_numbers:
            if str(num) in ch:
                print(f"Skipping {ch} - contains excluded number {num}")
                should_skip = True
                break
        if should_skip:
            continue
            
        if ch in std_ch_positions:
            pos = std_ch_positions[ch]
            available_positions.append((ch, pos))
            print(f"Adding {ch} to available list")
        else:
            print(f"Warning: {ch} not found in standard montage")
    
    print(f"Found {len(available_positions)} available auditory positions")
    
    new_positions = [pos for _, pos in available_positions]
    standard_names = [name for name, _ in available_positions]
    
    print("Standard montage names for these positions:")
    for name in standard_names:
        print(f"  {name}")
    
    return new_positions, standard_names
def train_loop_iteration(model, embed_fn, batch_points, batch_voltages, batch_normalized_voltages, norm_params_volt, optimizer):
    embedded_points = embed_fn(batch_points)
    predicted_normalized = model(embedded_points)
    
    loss = img2mse(predicted_normalized, batch_normalized_voltages)
    

    
    metrics = voltage_metrics(predicted_normalized, batch_normalized_voltages)
    
    optimizer.zero_grad()
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item(), metrics, predicted_normalized


def create_eeg_gs(args, normalized_train_points):
    """
    Create GS-style EEG model.

    normalized_train_points: [N, 4] = normalized [x, y, z, t]
    """

    centers = torch.unique(normalized_train_points[:, :3], dim=0)

    model = GaussianEEGField(
        centers=centers,
        hidden_dim=args.gs_hidden_dim,
        learn_centers=args.gs_learn_centers
    ).to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lrate
    )

    start = 0

    # GS does not use Fourier positional encoding.
    embed_fn = lambda x: x

    print(f"Created EEG-GS model with {centers.shape[0]} Gaussian centers")

    return model, optimizer, start, embed_fn


def load_voltage_data_from_fif(fif_file_path, duration_seconds=3):
    
    raw = mne.io.read_raw_fif(fif_file_path, preload=True)
    sfreq = raw.info['sfreq']
    
    n_samples = int(duration_seconds * sfreq)
    
    data, times = raw[:, :n_samples]
    
    positions = []
    for ch in raw.info['chs']:
        positions.append([ch['loc'][0], ch['loc'][1], ch['loc'][2]])
    
    points_list = []
    voltages_list = []
    
    for t_idx, time_point in enumerate(times):
        for e_idx, position in enumerate(positions):
            x, y, z = position
            points_list.append([x, y, z, time_point])
            voltages_list.append([data[e_idx, t_idx]])
    
    points = torch.tensor(points_list, dtype=torch.float32)
    voltages = torch.tensor(voltages_list, dtype=torch.float32)
    
    return points, voltages, raw



def voltage_metrics(pred, target):
 


    # Convert to numpy for sklearn metrics
    pred_np = pred.cpu().detach().numpy().flatten()
    target_np = target.cpu().detach().numpy().flatten()
    epsilon = 1e-22
    # Basic metrics using sklearn
    mse = mean_squared_error(target_np, pred_np)
    mae = mean_absolute_error(
        target_np, 
        pred_np
    )
    var = np.var(target_np)
    r2 = r2_score(target_np, pred_np)
    
    # Small constant to prevent division by zero
    rel_error = np.mean(np.abs(pred_np - target_np) / (np.abs(target_np) + epsilon))
    
    pcc = np.corrcoef(target_np, pred_np)[0, 1]
    
    signal_power = np.mean(target_np ** 2)
    noise_power = np.mean((target_np - pred_np) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + epsilon))
    
    
    nmse = mse / var
 

    return {'mse': float(mse),'mae': float(mae),'rel_error': float(rel_error),'r2': float(r2), 'pcc': float(pcc), 'snr': float(snr),'nmse': float(nmse) }







def setup_logger(args):
    log_dir = os.path.join(args.basedir, args.expname)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training_log.txt')
    
    with open(log_file, 'w') as f:
        f.write(f"training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write("\n" + "="*50 + "\n\n")
    
    return log_file

def log_metrics(log_file, iteration, metrics, loss, phase="TRAIN"):
    #Write metrics to log file

    with open(log_file, 'a') as f:
        f.write(f"\n[{phase}] Epoch: {iteration}\n")
        current_mse = metrics['mse'][-1] if isinstance(metrics['mse'], list) else metrics['mse']
        current_mae = metrics['mae'][-1] if isinstance(metrics['mae'], list) else metrics['mae']
        current_rel_error = metrics['rel_error'][-1] if isinstance(metrics['rel_error'], list) else metrics['rel_error']
        current_r2 = metrics['r2'][-1] if isinstance(metrics['r2'], list) else metrics['r2']
        current_pcc = metrics['pcc'][-1] if isinstance(metrics['pcc'], list) else metrics['pcc']
        current_snr = metrics['snr'][-1] if isinstance(metrics['snr'], list) else metrics['snr']
        current_nmse = metrics['nmse'][-1] if isinstance(metrics['nmse'], list) else metrics['nmse']
        
        # Write to file
        f.write(f"MSE: {current_mse:.15f}\n")
        f.write(f"MAE: {current_mae:.15f}\n")
        f.write(f"Relative Error: {current_rel_error:.15f}\n")
        f.write(f"R2: {current_r2:.15f}\n")
        f.write(f"PCC: {current_pcc:.15f}\n")
        f.write(f"SNR: {current_snr:.15f}\n")
        f.write(f"NMSE: {current_nmse:.15f}\n")
        f.write(f"Loss: {loss:.15f}\n")
        

            
        # Print to console
        print(f"MSE: {current_mse:.15f}")
        print(f"MAE: {current_mae:.15f}")
        print(f"Relative Error: {current_rel_error:.15f}")
        print(f"R2: {current_r2:.15f}")
        print(f"PCC: {current_pcc:.15f}")
        print(f"SNR: {current_snr:.15f}")
        print(f"NMSE: {current_nmse:.15f}")
        print(f"Loss: {loss:.15f}")
        






def generate_synthetic_data_first_window(model, embed_fn, raw, train_coord_norm_params, train_voltage_norm_params, window_start, window_duration):

    print(f"Generating synthetic data for first window ({window_start}s to {window_start+window_duration}s)")

    from mne.io.constants import FIFF
    
    lpa_coords = None
    nasion_coords = None
    rpa_coords = None
    
    for dig_point in raw.info['dig']:
        if dig_point['kind'] == FIFF.FIFFV_POINT_CARDINAL:
            if dig_point['ident'] == FIFF.FIFFV_POINT_LPA:
                lpa_coords = dig_point['r'].tolist()
            elif dig_point['ident'] == FIFF.FIFFV_POINT_NASION:
                nasion_coords = dig_point['r'].tolist()
            elif dig_point['ident'] == FIFF.FIFFV_POINT_RPA:
                rpa_coords = dig_point['r'].tolist()
    
    print(f"Original fiducials - LPA: {lpa_coords}, Nasion: {nasion_coords}, RPA: {rpa_coords}")
    
    data_window, times_window = raw[:, :]
    n_times = len(times_window)
    
    original_positions = []
    original_names = []
    
    for ch in raw.info['chs']:
        original_positions.append([ch['loc'][0], ch['loc'][1], ch['loc'][2]])
        original_names.append(ch['ch_name'])
    
    new_positions, new_names = add_auditory_electrodes(raw)
    
    points_list = []
    electrode_time_indices = [] 
    
    for t_idx, time_point in enumerate(times_window):
        absolute_time = time_point + window_start
        for e_idx, position in enumerate(new_positions):
            x, y, z = position
            points_list.append([x, y, z, absolute_time])
            electrode_time_indices.append((e_idx, t_idx))
    
    points = torch.tensor(points_list, dtype=torch.float32)
    
    normalized_points = normalize_coordinates(points, train_coord_norm_params)
    
    model.eval()
    
    print(f"Generating predictions for {len(normalized_points)} points ")
    print(f"Using exactly {n_times} time points for window starting at {window_start}s")
    
    with torch.no_grad():
        batch_size = 10000
        all_predictions = []
        
        for i in range(0, len(normalized_points), batch_size):
            batch = normalized_points[i:i+batch_size]
            embedded_points = embed_fn(batch)
            predicted_normalized = model(embedded_points)
            predictions_batch = denormalize_voltage(predicted_normalized, train_voltage_norm_params)
            all_predictions.append(predictions_batch)
        
        if len(all_predictions) > 1:
            all_predictions = torch.cat(all_predictions)
        else:
            all_predictions = all_predictions[0]
    
    n_new_electrodes = len(new_positions)
    new_data = np.zeros((n_new_electrodes, n_times))
    
    for i, (e_idx, t_idx) in enumerate(electrode_time_indices):
        new_data[e_idx, t_idx] = all_predictions[i].cpu().numpy()
    
    new_info = mne.create_info(ch_names=new_names, sfreq=raw.info['sfreq'], ch_types=['eeg']*n_new_electrodes)
    
    synthetic_raw = mne.io.RawArray(new_data, new_info)
    
    # Create dictionary with all electrode positions (original + new)
    all_ch_pos = {}
    
    for i, pos in enumerate(original_positions):
        x, y, z = pos[0], pos[1], pos[2]
        all_ch_pos[original_names[i]] = np.array([x, y, z])
    
    for i, pos in enumerate(new_positions):
        x, y, z = pos[0], pos[1], pos[2]
        all_ch_pos[new_names[i]] = np.array([x, y, z])
    
    digMontage = mne.channels.make_dig_montage(
        ch_pos=all_ch_pos,
        nasion=nasion_coords,  
        lpa=lpa_coords,        
        rpa=rpa_coords,        
        coord_frame='head'
    )
    
    print(f"\nGenerated data for {n_new_electrodes} new electrodes")
    print(f"Window time range: {window_start}s to {window_start + window_duration}s")
    
    print("\nelectrode positions:")
    print(f"total original electrodes: {len(original_positions)}")
    print(f"total new electrodes: {len(new_positions)}")
    for i, (pos, name) in enumerate(zip(original_positions, original_names)):
        print("{:<10} {:<10} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            "Original", name, pos[0], pos[1], pos[2]))
    
    # Print new positions
    for i, (pos, name) in enumerate(zip(new_positions, new_names)):
        print("{:<10} {:<10} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            "New", name, pos[0], pos[1], pos[2]))
    synthetic_raw.set_montage(digMontage)
    return synthetic_raw, digMontage, new_positions, new_names



def load_voltage_data_from_fif_window(raw_window, window_start, window_duration):

   # Load voltage data from a pre-cropped MNE Raw object for a specific window
    

    print(f"Loading data from window starting at {window_start}s with duration {window_duration}s")
    
    data, times = raw_window[:, :]
  
    adjusted_times = times + window_start
    
    positions = []
    for ch in raw_window.info['chs']:
        positions.append([ch['loc'][0], ch['loc'][1], ch['loc'][2]])
    
    points_list = []
    voltages_list = []
    
    for t_idx, time_point in enumerate(adjusted_times):
        for e_idx, position in enumerate(positions):
            x, y, z = position
            points_list.append([x, y, z, time_point])
            voltages_list.append([data[e_idx, t_idx]])
    
    points = torch.tensor(points_list, dtype=torch.float32)
    voltages = torch.tensor(voltages_list, dtype=torch.float32)
    
    print(f"Loaded {len(points)} data points from window")
    
    return points, voltages, raw_window

def generate_synthetic_data_subsequent_window(model, embed_fn, raw, train_coord_norm_params, 
                                             train_voltage_norm_params, window_start, window_duration,
                                             new_electrode_positions, new_electrode_names):

    #Generate synthetic electrode data for subsequent windows using the same electrode positions
        
    #Returns: mne.io.RawArray: Raw object with synthetic data for this window

    print(f"Generating synthetic data for subsequent window ({window_start}s to {window_start+window_duration}s)")

    
    data_window, times_window = raw[:, :]

    n_times = len(times_window)
    
    points_list = []
    electrode_time_indices = []  
    
    for t_idx, time_point in enumerate(times_window):
        absolute_time = time_point + window_start
        for e_idx, position in enumerate(new_electrode_positions):
            x, y, z = position
            points_list.append([x, y, z, absolute_time])
            electrode_time_indices.append((e_idx, t_idx))
    
    points = torch.tensor(points_list, dtype=torch.float32)
    
    normalized_points = normalize_coordinates(points, train_coord_norm_params)
    
    model.eval()
    
    print(f"Generating predictions for {len(normalized_points)} points ")
    print(f"Using exactly {n_times} time points for window starting at {window_start}s")
    
    with torch.no_grad():
        batch_size = 10000
        all_predictions = []
        
        for i in range(0, len(normalized_points), batch_size):
            batch = normalized_points[i:i+batch_size]
            embedded_points = embed_fn(batch)
            predicted_normalized = model(embedded_points)
            predictions_batch = denormalize_voltage(predicted_normalized, train_voltage_norm_params)
            all_predictions.append(predictions_batch)
        
        if len(all_predictions) > 1:
            all_predictions = torch.cat(all_predictions)
        else:
            all_predictions = all_predictions[0]
    
    n_new_electrodes = len(new_electrode_positions)
    new_data = np.zeros((n_new_electrodes, n_times))
    
    for i, (e_idx, t_idx) in enumerate(electrode_time_indices):
        new_data[e_idx, t_idx] = all_predictions[i].cpu().numpy()
    
    new_info = mne.create_info(ch_names=new_electrode_names, sfreq=raw.info['sfreq'], ch_types=['eeg'] * n_new_electrodes)
    
    synthetic_raw = mne.io.RawArray(new_data, new_info)
    
    print(f"\nGenerated data for {n_new_electrodes} new electrodes")
    print(f"Window time range: {window_start}s to {window_start + window_duration}s")
    
    return synthetic_raw

def train_eeg_gs_sliding_window(fif_file_path, model_dir, window_size=3, overwrite_original=True):

    # Train EEG-GS model on EEG data from a .fif file using a sliding window approach
    # and add new GS-generated auditory electrodes to the original file.
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    
    print("\nStarting sliding window EEG-GS processing")
    
    raw = mne.io.read_raw_fif(fif_file_path, preload=True)
    
    total_duration = raw.times[-1]
    print(f"total recording duration: {total_duration} seconds")
    
    n_windows = int(np.ceil(total_duration / window_size))
    print(f"Number of windows to process: {n_windows}")
    
    parser = config_parser()
    args = parser.parse_args([])  
    args.expname = "eeg_gs"
    args.basedir = model_dir
    
    original_raw_windows = []
    synthetic_raw_windows = []
    
    digMontage = None
    new_electrode_positions = None
    new_electrode_names = None
    
    for window_idx in range(n_windows):
        window_start = window_idx * window_size
        window_end = min(window_start + window_size, total_duration)
        window_duration = window_end - window_start
        
        print(f"\nprocessing window {window_idx + 1}/{n_windows}")
        print(f"time range: {window_start}s to {window_end}s")
        
        log_file = setup_logger(args)
        
        
        output_dir = os.path.join(args.basedir, args.expname)
        
        if window_idx < n_windows - 1:  
            raw_window = raw.copy().crop(tmin=window_start, tmax=window_end-(1.0/raw.info['sfreq']))
        else:  
            raw_window = raw.copy().crop(tmin=window_start, tmax=window_end)
            
        original_raw_windows.append(raw_window) 
        
        train_points, train_voltages, _ = load_voltage_data_from_fif_window(raw_window, window_start, window_duration)

        train_coord_norm_params = get_coordinate_normalization_params(train_points)
        train_voltages_normalized, train_voltage_norm_params = normalize_voltage(train_voltages)

        normalized_train_points = normalize_coordinates(train_points, train_coord_norm_params)

        model, optimizer, start, embed_fn = create_eeg_gs(
            args,
            normalized_train_points
        )

        train_dataset = VoltageDataset(normalized_train_points, train_voltages)
        
        generator = torch.Generator(device='cuda')
        train_loader = DataLoader(train_dataset, batch_size=args.N_rand, shuffle=True, generator=generator)
        
        n_epochs = 10
        patience = 10
        best_train_loss = float('inf')
        epochs_without_improvement = 0
        train_losses_per_epoch = []
        train_metrics_per_epoch = {'mse': [], 'mae': [], 'rel_error': [], 'r2': [], 'pcc': [], 'snr': [], 'nmse': []}
        
        print(f"training on window {window_idx + 1}/{n_windows} for {n_epochs} epochs")
        for epoch in range(n_epochs):
            model.train()
            train_loss_epoch = 0
            train_metrics_epoch = {'mse': 0, 'mae': 0, 'rel_error': 0, 'r2': 0, 'pcc': 0, 'snr': 0, 'nmse': 0}
            batch_count = 0
            
            for batch_points, batch_voltages, batch_normalized_voltages, norm_params in train_loader:
                batch_points = batch_points.to(device)
                batch_voltages = batch_voltages.to(device)
                batch_normalized_voltages = batch_normalized_voltages.to(device)
                
                loss, metrics, _ = train_loop_iteration(model, embed_fn, batch_points, batch_voltages, batch_normalized_voltages, norm_params, optimizer)
                
                train_loss_epoch += loss
                for key in metrics:
                    train_metrics_epoch[key] += metrics[key]
                
                batch_count += 1

            train_losses_per_epoch.append(train_loss_epoch / batch_count)
            for key in train_metrics_epoch:
                train_metrics_per_epoch[key].append(train_metrics_epoch[key] / batch_count)
            loss_epochTrain = train_loss_epoch / batch_count
            
            log_metrics(log_file, epoch, train_metrics_per_epoch, loss_epochTrain, "TRAIN")
            
            if train_losses_per_epoch[-1] < best_train_loss:
                best_train_loss = train_losses_per_epoch[-1]
                epochs_without_improvement = 0
                path = os.path.join(args.basedir, args.expname, f'best_model_window_{window_idx}.tar')
                
                os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)

                torch.save({
                    "model_type": "eeg_gs",
                    "window_idx": window_idx,
                    "epoch": epoch,
                    "network_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_coord_norm_params": train_coord_norm_params,
                    "train_voltage_norm_params": train_voltage_norm_params,
                    "best_train_loss": best_train_loss,
                }, path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    with open(log_file, 'a') as f:
                        f.write(f"\nEarly stopping at epoch {epoch}")
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
        
        print(f"training complete for window {window_idx + 1}. generating new electrode data")
        
        if window_idx == 0:
            synthetic_raw, digMontage, new_electrode_positions, new_electrode_names = generate_synthetic_data_first_window(
                model=model,
                embed_fn=embed_fn,
                raw=raw_window,
                train_coord_norm_params=train_coord_norm_params,
                train_voltage_norm_params=train_voltage_norm_params,
                window_start=window_start,
                window_duration=window_duration
            )
            synthetic_raw_windows.append(synthetic_raw)
        else:
            synthetic_raw = generate_synthetic_data_subsequent_window(
                model=model,
                embed_fn=embed_fn,
                raw=raw_window,
                train_coord_norm_params=train_coord_norm_params,
                train_voltage_norm_params=train_voltage_norm_params,
                window_start=window_start,
                window_duration=window_duration,
                new_electrode_positions=new_electrode_positions,
                new_electrode_names=new_electrode_names
            )
            synthetic_raw_windows.append(synthetic_raw)

    print("\nconcatenating all windows")
    original_data_all = mne.concatenate_raws(original_raw_windows)
    
    synthetic_data_all = mne.concatenate_raws(synthetic_raw_windows)
    
    print(f"original data time points: {len(original_data_all.times)}")
    print(f"synthetic data time points: {len(synthetic_data_all.times)}")
    
    if len(original_data_all.times) != len(synthetic_data_all.times):
        print(f"WARNING: Time point mismatch: {len(original_data_all.times)} vs {len(synthetic_data_all.times)}")
        if len(synthetic_data_all.times) > len(original_data_all.times):
            print("trimming synthetic data to match original ")
            synthetic_data_all = synthetic_data_all.crop(tmin=0, tmax=original_data_all.times[-1])
        elif len(original_data_all.times) > len(synthetic_data_all.times):
            print("trimming original data to match synthetic ")
            original_data_all = original_data_all.crop(tmin=0, tmax=synthetic_data_all.times[-1])
    
    if overwrite_original:
        raw_original = mne.io.read_raw_fif(fif_file_path, preload=True)
        
        if len(raw_original.times) != len(synthetic_data_all.times):
            print(f"WARNING: Time point mismatch between original file and synthetic data")
            print(f"original: {len(raw_original.times)}, Synthetic: {len(synthetic_data_all.times)}")
            synthetic_data_all = synthetic_data_all.crop(tmin=0, tmax=raw_original.times[-1])
        
        raw_original.add_channels([synthetic_data_all], force_update_info=True)
        
        raw_original.set_montage(digMontage, on_missing='ignore')
        
        raw_original.save(fif_file_path, overwrite=True)
        
        print("\nall windows processed successfully.")
        print(f"original file updated with synthetic data: {fif_file_path}")
        return raw_original
    else:
        output_file_path = fif_file_path.replace('-raw.fif', '-generated-raw.fif')
        combined_raw = original_data_all.copy()
        combined_raw.add_channels([synthetic_data_all], force_update_info=True)
        combined_raw.set_montage(digMontage, on_missing='ignore')
        combined_raw.save(output_file_path, overwrite=True)
        
        print("\nall windows processed successfully.")
        print(f"final output saved to {output_file_path}")
        return combined_raw


def config_parser():
    """Setup configuration parser for EEG-GS"""
    parser = configargparse.ArgumentParser()

    # Basic file paths
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, default="eeg_gs", help='experiment name')
    parser.add_argument("--basedir", type=str, default='./output', help='where to store checkpoints and logs')

    # GS architecture
    parser.add_argument("--gs_hidden_dim", type=int, default=64, help='hidden dimension for GS temporal network')
    parser.add_argument("--gs_learn_centers", action='store_true', help='allow Gaussian centers to move during training')

    # Training options
    parser.add_argument("--N_rand", type=int, default=32, help='batch size')
    parser.add_argument("--lrate", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')

    return parser


if __name__ == '__main__':
    base_path = r"C:\NBF_DATA\brennan2019_processed"
    
    for patient_num in range(1,49):
        patient_id = f"S{patient_num:02d}"
        patient_dir = os.path.join(base_path, patient_id)
        
        if not os.path.exists(patient_dir):
            print(f"Patient directory {patient_dir} does not exist. Skipping ")
            continue
        
        fif_file_path = os.path.join(patient_dir, "meg-sr120-hp0-raw.fif")
        
        if not os.path.exists(fif_file_path):
            print(f"Input file {fif_file_path} does not exist. Skipping ")
            continue
        
        try:
            temp_raw = mne.io.read_raw_fif(fif_file_path, preload=False)
            num_electrodes = len(temp_raw.ch_names)
            print(f"Patient {patient_id} has {num_electrodes} electrodes")
            
            if num_electrodes > 62: #making sure that we are not proccessing a file from the brennan2019 dataset that has already been proccessed by our model
                print(f"Patient {patient_id} has more than 62 electrodes ({num_electrodes}). Skipping ")
                continue
        except Exception as e:
            print(f"Error checking electrodes for patient {patient_id}: {str(e)}")
            continue
        
        model_dir = os.path.join(patient_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"\nProcessing patient {patient_id} ")
        
        try:
            combined_raw = train_voltage_nerf_sliding_window(
                fif_file_path, model_dir, window_size=3, overwrite_original=True)
            
            print(f"\nTraining Complete for patient {patient_id}")
            print(f"Original file updated: {fif_file_path}")
        except Exception as e:
            print(f"Error processing patient {patient_id}: {str(e)}")
            continue
    

