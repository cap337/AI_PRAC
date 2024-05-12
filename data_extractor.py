import pandas as pd
import numpy as np

def extract_features_and_labels(csv_file):
    # Load the dataset
    data = pd.read_csv(csv_file)

    # Assume the last column is the label, and the rest are features
    features_np = data.iloc[:, :-1].values  # All rows, all columns except the last
    labels_np = data.iloc[:, -1].values  # All rows, last column

    return features_np, labels_np