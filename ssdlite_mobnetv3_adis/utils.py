from __future__ import annotations
import os
import zipfile
from tqdm import tqdm
import torch.nn as nn

def unzip_file(zip_path: str, target_dir: str) -> None:
    """
    Unzips the specified zip file into the target directory with a progress bar.

    Parameters:
    - zip_path (str): The path to the zip file to be unzipped.
    - target_dir (str): The directory where the unzipped files will be stored.
    """
    # Ensure the target directory exists; create it if it doesn't
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of all files and directories in the zip
        info_list = zip_ref.infolist()
        # Calculate total uncompressed size for the progress bar
        total_size = sum(zinfo.file_size for zinfo in info_list)
        
        # Create progress bar with total size in bytes, scaled to KB/MB/GB as needed
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Unzipping") as pbar:
            for zinfo in info_list:
                # Extract the file or directory
                zip_ref.extract(zinfo, target_dir)
                # Update progress bar by the file's uncompressed size
                pbar.update(zinfo.file_size)
    
    print(f"Unzipped {zip_path} to {target_dir}")
    
    # Optionally, you can remove the zip file after extraction
    os.remove(zip_path)
    print(f"Removed zip file: {zip_path}")
    
    
def replace_activation_function(module: nn.Module, activation_fn) -> None:
    for name, child in module.named_children():
        # catch both ReLU and ReLU6
        if isinstance(child, (nn.ReLU, nn.ReLU6)):
            # replace the activation function with the new one
            setattr(module, name, activation_fn)
        else:
            # recursively call the function for child modules
            replace_activation_function(child, activation_fn)