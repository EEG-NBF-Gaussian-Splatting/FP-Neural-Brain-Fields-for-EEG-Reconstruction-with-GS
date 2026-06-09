import mne
import os
from pathlib import Path

base_dir = r"C:\NBF_DATA\brennan2019_processed"

# Loop through patients S01 to S48
for patient_num in range(1, 49):
    # Format patient directory name (S01, S02, etc.)
    patient_dir = f"S{patient_num:02d}"
    
    patient_path = os.path.join(base_dir, patient_dir)
    fif_path = os.path.join(patient_path, "meg-sr120-hp0-raw.fif")
    
    if not os.path.exists(patient_path):
        print(f"skipping {patient_dir}: directory does not exist")
        continue
    
    if not os.path.exists(fif_path):
        print(f"skipping {patient_dir}: file does not exist")
        continue
    
    try:
        print(f"Processing {patient_dir}")
        
        raw = mne.io.read_raw_fif(fif_path, preload=True)
        
        raw.filter(l_freq=0.5, h_freq=59.5, fir_design='firwin')
        
        # Overwrite the original file with the filtered version
        raw.save(fif_path, overwrite=True)
        
        print(f"successfully processed {patient_dir}")
        
    except Exception as e:
        print(f"error processing {patient_dir}: {str(e)}")
        