#!/usr/bin/env python3
"""
Detailed training monitor showing epoch and loss information
"""

import os
import time
import numpy as np
import subprocess
from pathlib import Path
import psutil

def get_training_process_info():
    """Get information about the training process"""
    try:
        result = subprocess.run(['pgrep', '-f', 'model_training'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pid = result.stdout.strip()
            process = psutil.Process(int(pid))
            return {
                'pid': pid,
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'create_time': process.create_time()
            }
    except:
        pass
    return None

def read_tensorboard_logs(log_dir):
    """Read TensorBoard logs to extract training information"""
    try:
        event_files = list(Path(log_dir).glob("events.out.tfevents.*"))
        if event_files:
            latest_file = max(event_files, key=os.path.getmtime)
            file_size = os.path.getsize(latest_file)
            mod_time = time.ctime(os.path.getmtime(latest_file))
            return {
                'latest_file': latest_file.name,
                'file_size': file_size,
                'mod_time': mod_time
            }
    except:
        pass
    return None

def estimate_epoch_from_time(start_time, current_time):
    """Estimate current epoch based on time elapsed"""
    elapsed_minutes = (current_time - start_time) / 60
    # Assuming ~1 epoch per minute based on previous observations
    estimated_epoch = int(elapsed_minutes)
    return max(1, estimated_epoch)

def monitor_training():
    """Main monitoring function"""
    
    log_dir = Path("baseline/work_dir/unet_3class")
    
    print("🔍 Detailed Training Monitor")
    print("=" * 60)
    
    while True:
        # Get process info
        process_info = get_training_process_info()
        
        if not process_info:
            print("❌ Training process not found")
            break
            
        # Calculate time elapsed
        elapsed_seconds = time.time() - process_info['create_time']
        elapsed_minutes = elapsed_seconds / 60
        elapsed_hours = elapsed_minutes / 60
        
        # Estimate current epoch
        estimated_epoch = estimate_epoch_from_time(process_info['create_time'], time.time())
        
        # Get TensorBoard info
        tb_info = read_tensorboard_logs(log_dir)
        
        # Display information
        print(f"📊 Epoch (estimated): {estimated_epoch}/100")
        print(f"⏱️  Time elapsed: {elapsed_hours:.1f}h {elapsed_minutes%60:.0f}m {elapsed_seconds%60:.0f}s")
        print(f"📈 Progress: {estimated_epoch}%")
        
        if tb_info:
            print(f"📝 Latest log: {tb_info['latest_file']}")
            print(f"📦 Log size: {tb_info['file_size']:,} bytes")
            print(f"🕐 Last update: {tb_info['mod_time']}")
        
        print(f"💻 CPU: {process_info['cpu_percent']:.1f}%")
        print(f"🧠 Memory: {process_info['memory_mb']:.1f} MB")
        
        # Estimate remaining time
        if estimated_epoch > 0:
            time_per_epoch = elapsed_minutes / estimated_epoch
            remaining_epochs = 100 - estimated_epoch
            remaining_minutes = remaining_epochs * time_per_epoch
            remaining_hours = remaining_minutes / 60
            
            print(f"⏰ Time per epoch: {time_per_epoch:.1f} minutes")
            print(f"🎯 Estimated completion: {remaining_hours:.1f} hours remaining")
            
            # Calculate completion time
            completion_time = time.time() + (remaining_minutes * 60)
            completion_str = time.ctime(completion_time)
            print(f"🏁 Expected finish: {completion_str}")
        
        print("-" * 60)
        print(f"🕐 {time.strftime('%H:%M:%S')}")
        print("=" * 60)
        
        time.sleep(30)  # Update every 30 seconds

if __name__ == "__main__":
    monitor_training() 