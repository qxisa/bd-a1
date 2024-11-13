import matplotlib.pyplot as plt
import pandas as pd
import sys
import subprocess

def create_visualization(df):
    # Example visualization
    plt.scatter(df['Pclass'], df['Survived'])
    plt.xlabel('Pclass')
    plt.ylabel('Survived')
    plt.title('Simple Visualization')
    plt.savefig('vis.png')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python vis.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = pd.read_csv(file_path)

    create_visualization(df)
    print("Image created Sucessfully.")

    # Call model.py script as subprocess
  