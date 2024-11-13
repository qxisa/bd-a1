import pandas as pd
from sklearn.cluster import KMeans
import sys
import subprocess

def k_means_algorithm(df):


    X = df[['Pclass', 'Survived']] # Selecting the features for clustering (Pclass and Survived) from the dataframe df

    kmeans = KMeans(n_clusters=3) #creating a KMeans object with 3 clusters 

    kmeans.fit(X)

    cluster_counts = pd.Series(kmeans.labels_).value_counts().sort_index() # Counting the number of data points in each cluster

    with open('k.txt', 'w') as f: # Writing the cluster counts to a file
        f.write(str(cluster_counts)) 

if __name__ == "__main__":
    if len(sys.argv) != 2: # Checking if the number of arguments is correct 
        sys.exit(1)
    
    file_path = sys.argv[1]# Reading the file path from the command line arguments
    df = pd.read_csv(file_path)

    k_means_algorithm(df)

    print("algorithm executed successfully")

    subprocess.run(["python3", "vis.py", file_path])