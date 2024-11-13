import pandas as pd
import sys
import subprocess

def exploratory_data_analysis(file_path):
    df = pd.read_csv(file_path)

    insights = [
        "1. There are {} rows and {} columns in the dataset.".format(df.shape[0], df.shape[1]),
        "2. The first five rows of the dataset:\n{}".format(df.head()),
        "3. The summary statistics of numerical columns:\n{}".format(df.describe())
    ]

    for i, insight in enumerate(insights):
        with open(f'eda-in-{i+1}.txt', 'w') as f:
            f.write(insight)

    print("Exploratory data analysis insights saved successfully.")

    subprocess.run(["python3", "dpre.py", file_path])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eda.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    exploratory_data_analysis(file_path)