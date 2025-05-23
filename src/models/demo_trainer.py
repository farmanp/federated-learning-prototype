"""
Demonstration of the local model trainer module for federated learning.
"""

import numpy as np
from src.utils.data_loader import load_data, preprocess_data
from src.models.trainer import train_local_model
from src.smc.paillier import PaillierCrypto, aggregate_encrypted_vectors, generate_keys
from loguru import logger

def main():
    """
    Demonstrate the model trainer module with a simple federated learning scenario.
    """
    print("============================================================")
    print("       Local Model Training for Federated Learning          ")
    print("============================================================")
    
    # 1. Set up the crypto system (typically done by the aggregator)
    print("\n1. Initializing cryptosystem...")
    crypto = PaillierCrypto(key_size=1024)  # Using smaller key for demo
    public_key, private_key = crypto.generate_keys()
    
    # 2. Simulate data for multiple parties
    print("\n2. Generating and partitioning data for 3 simulated parties...")
    # Generate a synthetic dataset
    all_data = load_data(
        file_path=None,  # Generate synthetic data
        n_samples=1000,
        n_features=10,
        n_classes=2,
        random_state=42
    )
    
    # Create 3 partitions by sampling rows (simulating distributed data)
    np.random.seed(42)
    n_samples = len(all_data)
    partition_indices = [
        np.random.choice(n_samples, size=n_samples//3, replace=False)
        for _ in range(3)
    ]
    
    data_party_dfs = [
        all_data.iloc[indices].reset_index(drop=True)
        for indices in partition_indices
    ]
    
    print(f"   Data partitioned: {[len(df) for df in data_party_dfs]} samples per party")
    
    # 3. Each party preprocesses their data and trains a local model
    print("\n3. Each party trains a local model:")
    
    model_weights = []
    accuracies = []
    
    for i, party_df in enumerate(data_party_dfs):
        print(f"\n   Data Party {i+1}:")
        # Preprocess the data
        X_train, X_test, y_train, y_test = preprocess_data(party_df)
        
        # Train a local model
        weights, accuracy, metrics = train_local_model(X_train, y_train, X_test, y_test)
        
        # Store the results
        model_weights.append(weights)
        accuracies.append(accuracy)
        
        print(f"   - Trained model with {metrics['n_train_samples']} samples")
        print(f"   - Local test accuracy: {accuracy:.4f}")
    
    # 4. Each party encrypts their model weights
    print("\n4. Each party encrypts their model weights...")
    encrypted_weights = [
        crypto.encrypt_vector(weights)
        for weights in model_weights
    ]
    
    # 5. Aggregator securely combines the encrypted model weights
    print("\n5. Aggregator performs secure aggregation on encrypted weights...")
    aggregated_encrypted = aggregate_encrypted_vectors(encrypted_weights)
    
    # 6. Aggregator decrypts the result
    print("\n6. Aggregator decrypts the aggregated result...")
    decrypted_weights = crypto.decrypt_vector(aggregated_encrypted)
    
    # For comparison, calculate the plain sum directly
    sum_weights = np.sum(model_weights, axis=0).tolist()
    
    print(f"\n   Average of local accuracies: {np.mean(accuracies):.4f}")
    print(f"   Local model weights (first 3 elements):")
    for i, weights in enumerate(model_weights):
        print(f"   - Party {i+1}: {[round(w, 4) for w in weights[:3]]}...")
    
    print(f"\n   Securely aggregated weights (first 3 elements): {[round(w, 4) for w in decrypted_weights[:3]]}...")
    print(f"   Plain sum weights (first 3 elements): {[round(w, 4) for w in sum_weights[:3]]}...")
    
    # Verify that the secure aggregation worked correctly
    is_correct = all(
        abs(expected - actual) < 1e-10  # Using small epsilon for floating point comparison
        for expected, actual in zip(sum_weights, decrypted_weights)
    )
    
    if is_correct:
        print("\n   SUCCESS: Secure aggregation produced the correct result!")
    else:
        print("\n   ERROR: Secure aggregation result doesn't match expected average.")
    
    print("\n7. In a real FL system, the aggregator would now:")
    print("   - Apply differential privacy noise to the aggregated weights")
    print("   - Distribute the new global model to all parties")
    print("   - Begin the next round of training")
    
    print("\n============================================================")

if __name__ == "__main__":
    main()
