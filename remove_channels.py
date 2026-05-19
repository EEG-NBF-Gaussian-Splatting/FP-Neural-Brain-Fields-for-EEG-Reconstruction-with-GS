import os
import mne

def clean_fif_file_keep_electrode(fif_file_path, keep_electrode="61", synthetic_threshold=62):
    
    #Cleans a .fif file by keeping only specified synthetic electrodes and removing the rest.

    print(f"\nProcessing: {fif_file_path}")
    
    raw = mne.io.read_raw_fif(fif_file_path, preload=True) #load all the record
    print(f"\nRaws: {raw}")
    all_channels = raw.ch_names #return list of all the channels names  
    print(f"\nAll channels: {all_channels}")

    
    keep_electrodes = []
    if "-" in str(keep_electrode):
        start, end = keep_electrode.split("-")
        keep_electrodes = [str(i) for i in range(int(start), int(end) + 1)]
    else:
        keep_electrodes = [str(keep_electrode)]
    
    channels_to_drop = []
    for ch in all_channels:
        if ch.isdigit() and int(ch) >= synthetic_threshold and ch not in keep_electrodes:
            channels_to_drop.append(ch)
    
    if not channels_to_drop:
        print("No synthetic electrodes found to drop.")
    else:
        print(f"Dropping synthetic electrodes (except {keep_electrodes}): {channels_to_drop}")
        raw.drop_channels(channels_to_drop)
    
    raw.save(fif_file_path, overwrite=True)
    print(f"File updated (overwritten): {fif_file_path}")




base_dir = "C:\\Users\\liron\\Desktop\\סמסטר 7\\Final Project\\FP-Neural-Brain-Fields-for-EEG-Reconstruction-with-GS"
patient_num = 1
patient_id = f"S{patient_num:02d}"
print(patient_id)
fif_path = os.path.join(base_dir, patient_id, "meg-sr120-hp0-raw.fif")
if os.path.exists(fif_path):
    try:
        # Keep electrodes 62-66 inclusive
        clean_fif_file_keep_electrode(fif_path, keep_electrode="62 - 66", synthetic_threshold=62)
    except Exception as e:
        print(f"Error processing {patient_id}: {str(e)}")
else:
    print(f"File not found: {fif_path}")


# for patient_num in range(1, 49):
#     patient_id = f"S{patient_num:02d}"
#     fif_path = os.path.join(base_dir, patient_id, "meg-sr120-hp0-raw.fif")
#     if os.path.exists(fif_path):
#         try:
#             # Keep electrodes 62-66 inclusive
#             clean_fif_file_keep_electrode(fif_path, keep_electrode="62 - 66", synthetic_threshold=62)
#         except Exception as e:
#             print(f"Error processing {patient_id}: {str(e)}")
#     else:
#         print(f"File not found: {fif_path}")
        

