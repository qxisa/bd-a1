import pandas as pd
import sys
import subprocess

def load_dataset(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python load.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    dataset = load_dataset(file_path)
    print("Dataset loaded successfully.")

    subprocess.run(["python3", "eda.py", file_path])