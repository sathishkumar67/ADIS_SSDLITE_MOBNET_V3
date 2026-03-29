"""Small utility helpers used across the project.

Currently provides `unzip_file` which extracts a zip archive with a
progress bar. Add other lightweight helpers here as needed.
"""

from __future__ import annotations
import os
import zipfile
from tqdm import tqdm
import torch.nn as nn
import threading
import time
try:
    import pynvml
except ImportError:
    pynvml = None


class GPUMonitor:
    """
    A context-managed GPU energy monitor that tracks power usage in a background thread.
    
    Usage:
        with GPUMonitor(device_index=0) as monitor:
            # code to track
            pass
        metrics = monitor.get_metrics()
        print(f"Avg Power: {metrics['avg_power_w']:.2f}W, Total Energy: {metrics['total_energy_j']:.2f}J")
    """
    def __init__(self, device_index=0, sampling_interval=0.1):
        self.device_index = device_index
        self.sampling_interval = sampling_interval
        self.power_samples = []
        self.running = False
        self.start_time = 0
        self.end_time = 0
        self._available = False
        self.handle = None

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self._available = True
            except Exception as e:
                print(f"Warning: GPU Monitoring not available: {e}")
        else:
            print("Warning: pynvml not installed. GPU Monitoring will be disabled.")

    def _sample_loop(self):
        while self.running:
            try:
                # nvmlDeviceGetPowerUsage returns power in milliwatts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.power_samples.append(power_mw / 1000.0) # Convert to Watts
            except Exception:
                pass
            time.sleep(self.sampling_interval)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        if not self._available:
            return
        self.power_samples = []
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.thread.join(timeout=1.0)
        self.end_time = time.time()

    def get_metrics(self) -> dict:
        """Returns a dictionary containing avg_power_w, total_energy_j, and duration_s."""
        duration = self.end_time - self.start_time if self.end_time > 0 else (time.time() - self.start_time)
        
        if not self.power_samples:
            return {"avg_power_w": 0.0, "total_energy_j": 0.0, "duration_s": duration}
        
        avg_power = sum(self.power_samples) / len(self.power_samples)
        total_energy = avg_power * duration # Joules = Watts * Seconds
        
        return {
            "avg_power_w": avg_power,
            "total_energy_j": total_energy,
            "duration_s": duration
        }



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