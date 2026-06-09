import os
import mne
import numpy as np

def update_channel_info(fif_file_path):

    print(f"\nprocessing file: {fif_file_path}")
    
    print("loading FIF file ")
    raw = mne.io.read_raw_fif(fif_file_path, preload=True)
    print(f"file loaded. Found {len(raw.ch_names)} channels")
    
    named_channels = ['T7', 'T8', 'TP7', 'TP8', 'FC5', 'FC6', 'FT7', 'FT8', 
                     'P7', 'P8', 'CP5', 'CP6', 'C5', 'C6', 'F7', 'F8', 'P5', 'P6']
    
    existing_named_channels = [ch for ch in named_channels if ch in raw.ch_names]
    print(f"found {len(existing_named_channels)} named channels to update: {existing_named_channels}")
    
    # Starting values
    start_ch_number = 62  # max original electrode + 1
    start_logno = 61      # max original logno + 1
    print(f"Will start numbering from channel {start_ch_number}, logno {start_logno}")
    

    mapping = {}
    for i, name in enumerate(named_channels):
        if name in raw.ch_names:
            mapping[name] = str(start_ch_number + i)
    
    print(f"channel mapping: {mapping}")
    
    if not mapping:
        print("no channels to rename. Skipping this file.")
        return
    
    print("renaming channels ")
    raw.rename_channels(mapping)
    print("channels renamed successfully")
    
    print("updating logno values ")
    for i, name in enumerate(mapping.values()):  
        idx = raw.ch_names.index(name)
        old_logno = raw.info['chs'][idx]['logno']
        new_logno = start_logno + i
        print(f"  Channel {name}: logno {old_logno} -> {new_logno}")
        raw.info['chs'][idx]['logno'] = new_logno
    
    if raw.info.get('dig') is not None:
        print(f"updating digitization points  (Found {len(raw.info['dig'])} points)")
        updated_count = 0
        for i, name in enumerate(mapping.values()):
            idx = raw.ch_names.index(name)
            ch_loc = raw.info['chs'][idx]['loc'][:3]
            
            for j, dig_point in enumerate(raw.info['dig']):
                if dig_point['kind'] == 3:  
                    if np.allclose(ch_loc, dig_point['r'], rtol=1e-5, atol=1e-8):
                        old_ident = dig_point['ident']
                        new_ident = start_ch_number + i
                        print(f"  Digitization point {j}: ident {old_ident} -> {new_ident}")
                        dig_point['ident'] = new_ident
                        updated_count += 1
                        break
        print(f"updated {updated_count} digitization points")
    else:
        print("no digitization points found in the file")
    
    print("saving modified file ")
    raw.save(fif_file_path, overwrite=True)
    print(f"file saved successfully: {fif_file_path}")

def process_all_fif_files(base_directory, fif_filename):

    print(f"\nstarting to process FIF files")
    print(f"base directory: {base_directory}")
    print(f"target filename: {fif_filename}")
    
    processed_count = 0
    error_count = 0
    found_count = 0
    
    print("scanning directories ")
    for root, dirs, files in os.walk(base_directory):
        if fif_filename in files:
            found_count += 1
            fif_path = os.path.join(root, fif_filename)
            print(f"\nFound file #{found_count}: {fif_path}")
            
            try:
                update_channel_info(fif_path)
                processed_count += 1
                print(f"successfully processed file #{processed_count}: {fif_path}")
            except Exception as e:
                error_count += 1
                print(f"ERROR processing file: {fif_path}")
                print(f"error details: {str(e)}")
                print(f"continuing to next file ")
    
    print(f"\nprocessing Summary")
    print(f"total files found: {found_count}")
    print(f"total files successfully processed: {processed_count}")
    print(f"total errors: {error_count}")
    if error_count > 0:
        print("some files had errors. Review the log output for details")
    else:
        print("all files processed successfully")

base_dir = r"C:\NBF_DATA\brennan2019_processed"
fif_filename = "meg-sr120-hp0-raw.fif"  

print(f"starting script at: {os.path.abspath(__file__)}")
process_all_fif_files(base_dir, fif_filename)
print("script execution complete.")
