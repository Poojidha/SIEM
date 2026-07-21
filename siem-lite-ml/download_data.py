import os
import shutil
from pathlib import Path

def setup_directories():
    base_path = Path(__file__).parent
    dirs = ['datasets', 'processed', 'models', 'results', 'figures']
    for d in dirs:
        (base_path / d).mkdir(parents=True, exist_ok=True)
    print("Directories created successfully.")

def copy_dataset():
    # We found the Wednesday dataset in the parent folder's MachineLearningCSV directory!
    base_path = Path(__file__).parent
    source_file = base_path.parent / 'MachineLearningCSV' / 'MachineLearningCVE' / 'DoS-Wednesday-no-metadata.parquet'
    dest_file = base_path / 'datasets' / 'Wednesday-WorkingHours.parquet'
    
    if not source_file.exists():
        print(f"Error: Could not find {source_file}. Make sure it exists.")
        return
        
    print(f"Copying {source_file} to {dest_file}...")
    shutil.copy(source_file, dest_file)
    print("Dataset ready!")

if __name__ == "__main__":
    setup_directories()
    copy_dataset()
