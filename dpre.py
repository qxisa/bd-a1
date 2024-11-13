import pandas as pd
import sys
import numpy as np
import subprocess
from sklearn.preprocessing import OneHotEncoder

def drop_duplicates(data):
    #remove duplicate rows
    return data.drop_duplicates()

def fill_missing(data):             #fill missing values in numeric columns with the column mean
    for col in data.select_dtypes(include=[np.number]).columns:
        data[col].fillna(data[col].mean(), inplace=True)
    return data

def convert_gender(data):           #convert sex column to numeric values
    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})
    return data

def encode_port(data):              #encoding 'Embarked' column using one-hot encoding which creates a new column for each category    
    encoder = OneHotEncoder(drop='first')
    port_encoded = encoder.fit_transform(data[['Embarked']])
    port_df = pd.DataFrame(port_encoded.toarray(), columns=encoder.get_feature_names_out(['Embarked'])) #create a dataframe from the encoded values
    return pd.concat([data, port_df], axis=1).drop(columns=['Embarked']) #concatenate the new dataframe with the original dataframe and drop the original 'Embarked' column

def create_family_size(data):       #create 'family_size' by combining 'sibsp' and 'parch', then drop those columns
    data['Family_Size'] = data['SibSp'] + data['Parch'] #create a new column 'Family_Size' by adding 'SibSp' and 'Parch'
    return data.drop(['SibSp', 'Parch'], axis=1)

def keep_important(data):
    return data.drop(['Name', 'Ticket', 'Cabin', 'Sex'], axis=1, errors='ignore')

def categorize_age(data):
    if 'Age' in data.columns:
        data['Age_Group'] = pd.cut(data['Age'], bins=[0, 11, 18, 59, data['Age'].max()]) 
    return data

def categorize_fare(data):
    data['Fare_Group'] = pd.qcut(data['Fare'], q=5, labels=False) #create 'Fare_Group' by dividing 'Fare' into 5 quantiles
    return data

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    data = drop_duplicates(data)
    data = fill_missing(data)
    
    data = convert_gender(data)
    data = encode_port(data)
    
    data = create_family_size(data)
    data = keep_important(data)
    
    data = categorize_age(data)
    data = categorize_fare(data)
    
    data.to_csv('processed_data.csv', index=False)
    
    subprocess.run(["python3", "model.py", "processed_data.csv"])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    preprocess_data(file_path)
