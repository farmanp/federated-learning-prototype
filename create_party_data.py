#!/usr/bin/env python3
"""
Script to create sample data for multiple parties based on the Iris dataset.
This divides the Iris dataset into parts for different parties.
"""
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# Combine features and target
df = pd.concat([X, y], axis=1)
print(f"Total dataset size: {len(df)} samples")

# Create directories if they don't exist
for i in range(1, 4):
    os.makedirs(f"data/party_{i}", exist_ok=True)
    print(f"Created directory: data/party_{i}")

# Split data for 3 parties
party_dfs = []
n_parties = 3
samples_per_party = len(df) // n_parties

for i in range(n_parties):
    if i < n_parties - 1:
        party_df = df.iloc[i * samples_per_party:(i + 1) * samples_per_party].copy()
    else:
        # Last party gets the remaining samples
        party_df = df.iloc[i * samples_per_party:].copy()
        
    party_dfs.append(party_df)
    print(f"Party {i+1} data size: {len(party_df)} samples")

# Save the data for each party
for i, party_df in enumerate(party_dfs):
    file_path = f"data/party_{i+1}/iris_party_{i+1}.csv"
    party_df.to_csv(file_path, index=False)
    print(f"Saved data to {file_path}")

print("Data creation complete!")
