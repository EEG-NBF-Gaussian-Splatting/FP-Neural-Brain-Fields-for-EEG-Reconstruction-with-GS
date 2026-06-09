import mne
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Path to the FIF file
fif_path = Path(r"C:\NBF_DATA_Backup\brennan2019_processed\S44\meg-sr120-hp0-raw.fif")

# Load FIF file
raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=True)

print("\n==============================")
print("BASIC FILE INFO")
print("==============================")
print(raw)

print("\n==============================")
print("CHANNEL NAMES")
print("==============================")
print(raw.ch_names)

print("\n==============================")
print("SAMPLING RATE")
print("==============================")
sfreq = raw.info["sfreq"]
print(f"Sampling frequency: {sfreq} Hz")

print("\n==============================")
print("NUMBER OF CHANNELS / SAMPLES")
print("==============================")
print(f"Number of channels: {len(raw.ch_names)}")
print(f"Number of samples: {raw.n_times}")
print(f"Duration: {raw.n_times / sfreq:.2f} seconds")
print(f"Duration: {raw.n_times / sfreq / 60:.2f} minutes")

print("\n==============================")
print("CHANNEL TYPES")
print("==============================")
channel_types = raw.get_channel_types()
print(channel_types)

print("\n==============================")
print("DATA MATRIX")
print("==============================")
data = raw.get_data()
times = raw.times

print(f"Data shape: {data.shape}")
print("Meaning: [channels, time_samples]")
print(f"Times shape: {times.shape}")

print("\nFirst 5 channels, first 10 samples:")
print(data[:5, :10])

print("\n==============================")
print("DATA STATISTICS")
print("==============================")
print(f"Minimum value: {np.min(data)}")
print(f"Maximum value: {np.max(data)}")
print(f"Mean value: {np.mean(data)}")
print(f"Standard deviation: {np.std(data)}")

print("\n==============================")
print("ANNOTATIONS")
print("==============================")
print(raw.annotations)

print("\n==============================")
print("EVENTS FROM ANNOTATIONS")
print("==============================")
events, event_id = mne.events_from_annotations(raw)
print("events shape:", events.shape)
print(events[:10])
print("event_id:", event_id)

print("\n==============================")
print("EXAMPLE WINDOW")
print("==============================")

# Example: first 3 seconds
start_sec = 0
window_sec = 3

start_sample = int(start_sec * sfreq)
end_sample = int((start_sec + window_sec) * sfreq)

eeg_window = data[:, start_sample:end_sample]
time_window = times[start_sample:end_sample]

print(f"Window start: {start_sec} sec")
print(f"Window length: {window_sec} sec")
print(f"Start sample: {start_sample}")
print(f"End sample: {end_sample}")
print(f"eeg_window shape: {eeg_window.shape}")
print(f"time_window shape: {time_window.shape}")

print("\n==============================")
print("PLOT ONE CHANNEL")
print("==============================")

# Plot first 10 seconds of the first channel
channel_index = 0
plot_duration_sec = 10
plot_end_sample = int(plot_duration_sec * sfreq)

plt.figure(figsize=(12, 4))
plt.plot(times[:plot_end_sample], data[channel_index, :plot_end_sample])
plt.xlabel("Time [sec]")
plt.ylabel("EEG value")
plt.title(f"Channel {raw.ch_names[channel_index]} - first {plot_duration_sec} seconds")
plt.grid(True)
plt.show()
