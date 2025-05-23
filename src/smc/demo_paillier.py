"""
Demonstration of the Paillier homomorphic encryption module for secure aggregation.
"""

import time
import numpy as np
from src.smc.paillier import PaillierCrypto, aggregate_encrypted_vectors

def main():
    """
    Demonstrate the Paillier encryption module with a simple example.
    """
    print("============================================================")
    print("   Paillier Homomorphic Encryption for Secure Aggregation   ")
    print("============================================================")
    
    # 1. Initialize the crypto system
    print("\n1. Initializing cryptosystem and generating keys...")
    start_time = time.time()
    crypto = PaillierCrypto(key_size=1024)  # Using smaller key for demo
    public_key, private_key = crypto.generate_keys()
    print(f"   Key generation took {time.time() - start_time:.2f} seconds")
    
    # 2. Simulate weight vectors from different data parties
    print("\n2. Simulating weight vectors from 3 data parties...")
    data_party_vectors = [
        [0.1, 0.2, 0.3, 0.4, 0.5],  # Data Party 1's weights
        [1.1, 1.2, 1.3, 1.4, 1.5],  # Data Party 2's weights
        [2.1, 2.2, 2.3, 2.4, 2.5],  # Data Party 3's weights
    ]
    
    for i, vector in enumerate(data_party_vectors):
        print(f"   Data Party {i+1}: {vector}")
    
    # Calculate the expected plain sum for verification
    expected_sum = [
        sum(vec[i] for vec in data_party_vectors)
        for i in range(len(data_party_vectors[0]))
    ]
    print(f"\n   Expected aggregate (plaintext sum): {expected_sum}")
    
    # 3. Each data party encrypts their vector
    print("\n3. Each data party encrypts their weight vector...")
    start_time = time.time()
    encrypted_vectors = [
        crypto.encrypt_vector(vector)
        for vector in data_party_vectors
    ]
    print(f"   Encryption took {time.time() - start_time:.2f} seconds")
    
    # 4. Aggregator securely combines the encrypted vectors
    print("\n4. Aggregator performs secure aggregation on encrypted vectors...")
    start_time = time.time()
    aggregated_encrypted = aggregate_encrypted_vectors(encrypted_vectors)
    print(f"   Aggregation took {time.time() - start_time:.2f} seconds")
    
    # 5. Aggregator decrypts the result
    print("\n5. Aggregator decrypts the aggregated result...")
    start_time = time.time()
    decrypted_result = crypto.decrypt_vector(aggregated_encrypted)
    print(f"   Decryption took {time.time() - start_time:.2f} seconds")
    print(f"   Decrypted result: {[round(x, 5) for x in decrypted_result]}")
    
    # 6. Verify that the secure aggregation worked correctly
    print("\n6. Verifying results...")
    is_correct = all(
        abs(expected - actual) < 1e-10  # Using small epsilon for floating point comparison
        for expected, actual in zip(expected_sum, decrypted_result)
    )
    
    if is_correct:
        print("   SUCCESS: Secure aggregation produced the correct result!")
    else:
        print("   ERROR: Secure aggregation result doesn't match expected sum.")
    
    print("\n============================================================")

if __name__ == "__main__":
    main()
