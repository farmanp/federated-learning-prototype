# Paillier Encryption Module for Secure Aggregation

This document provides detailed information about the Paillier homomorphic encryption module 
in the Federated Learning Prototype system, which is used for secure aggregation of model updates.

## Overview

Secure Multiparty Computation (SMC) is a critical component of privacy-preserving federated learning. 
The `paillier.py` module implements the Paillier homomorphic encryption scheme, which allows 
computations on encrypted data—specifically, it permits the addition of encrypted values without 
decryption.

In federated learning, this enables the aggregation of model updates (such as gradients or weights) 
from multiple data parties without revealing their individual contributions, enhancing privacy 
while maintaining utility.

## Module Location

`src/smc/paillier.py`

## Features

- **Key Generation and Management**
  - Generate secure public/private keypairs
  - Set existing keys for reuse across sessions

- **Encryption/Decryption Operations**
  - Encrypt individual floating-point values
  - Encrypt vectors (lists of floats) for model parameters
  - Decrypt encrypted values or vectors

- **Secure Aggregation**
  - Add encrypted vectors element-wise without decrypting
  - Preserve homomorphic properties during aggregation

- **Error Handling**
  - Validate key presence and compatibility
  - Check vector dimensions for aggregation
  - Provide meaningful error messages through logging

## API Reference

### `PaillierCrypto` Class

The main class that encapsulates Paillier cryptographic operations.

```python
class PaillierCrypto:
    def __init__(self, key_size: int = 2048):
        """Initialize with a specific key size (default: 2048 bits)."""
        
    def generate_keys(self) -> Tuple[PublicKey, PrivateKey]:
        """Generate a new keypair."""
        
    def set_keys(self, public_key=None, private_key=None):
        """Set existing keys instead of generating new ones."""
        
    def encrypt_value(self, value: float) -> EncryptedNumber:
        """Encrypt a single float value."""
        
    def encrypt_vector(self, vector: List[float]) -> List[EncryptedNumber]:
        """Encrypt a vector of float values."""
        
    def decrypt_value(self, encrypted_value: EncryptedNumber) -> float:
        """Decrypt a single encrypted value."""
        
    def decrypt_vector(self, encrypted_vector: List[EncryptedNumber]) -> List[float]:
        """Decrypt a vector of encrypted values."""
```

### Convenience Functions

For simpler usage without instantiating the class:

```python
def generate_keys(key_size: int = 2048) -> Tuple[PublicKey, PrivateKey]:
    """Generate a new keypair."""
    
def encrypt_vector(vector: List[float], public_key: PublicKey) -> List[EncryptedNumber]:
    """Encrypt a vector using the provided public key."""
    
def decrypt_vector(encrypted_vector: List[EncryptedNumber], private_key: PrivateKey) -> List[float]:
    """Decrypt a vector using the provided private key."""
    
def aggregate_encrypted_vectors(encrypted_vectors: List[List[EncryptedNumber]]) -> List[EncryptedNumber]:
    """Aggregate multiple encrypted vectors element-wise."""
```

## Usage in Federated Learning

In the federated learning context, the Paillier module supports the following workflow:

1. **Setup Phase**: 
   - Aggregator generates keypair and shares public key with data parties
   - Data parties use the public key to encrypt their model updates

2. **Training Phase**: 
   - Each data party trains a local model on private data
   - Local model updates are encrypted using the public key
   - Encrypted updates are sent to the aggregator

3. **Aggregation Phase**: 
   - Aggregator receives encrypted updates from all parties
   - Securely aggregates updates without seeing individual values
   - Decrypts the aggregated result using the private key
   - The decrypted aggregate is used to update the global model

## Example Usage

### Basic Encryption/Decryption

```python
from src.smc.paillier import PaillierCrypto

# Initialize and generate keys
crypto = PaillierCrypto(key_size=2048)
public_key, private_key = crypto.generate_keys()

# Encrypt a value
original_value = 3.14159
encrypted_value = crypto.encrypt_value(original_value)

# Decrypt a value
decrypted_value = crypto.decrypt_value(encrypted_value)
assert abs(original_value - decrypted_value) < 1e-10
```

### Secure Aggregation

```python
from src.smc.paillier import PaillierCrypto, aggregate_encrypted_vectors

# Initialize cryptosystem
crypto = PaillierCrypto()
public_key, private_key = crypto.generate_keys()

# Data parties' model weights
party1_weights = [0.1, 0.2, 0.3]
party2_weights = [1.1, 1.2, 1.3]
party3_weights = [2.1, 2.2, 2.3]

# Each party encrypts their weights
encrypted_weights = [
    crypto.encrypt_vector(party1_weights),
    crypto.encrypt_vector(party2_weights),
    crypto.encrypt_vector(party3_weights),
]

# Aggregator securely aggregates the encrypted weights
aggregated_encrypted = aggregate_encrypted_vectors(encrypted_weights)

# Aggregator decrypts the result
aggregated_weights = crypto.decrypt_vector(aggregated_encrypted)

# For comparison, calculate the plain sum directly
expected_sum = [
    party1_weights[i] + party2_weights[i] + party3_weights[i]
    for i in range(len(party1_weights))
]

# Verify the result
assert all(abs(expected - actual) < 1e-10 
           for expected, actual in zip(expected_sum, aggregated_weights))
```

## Security Considerations

- **Key Size**: Default key size is 2048 bits, which provides strong security. For production use, 3072 or 4096 bits may be preferred.
- **Key Management**: In real-world deployments, proper key management (e.g., secure storage, rotation) is crucial.
- **Performance**: Paillier encryption is computationally intensive. For very large models, consider optimization techniques or batching.
- **Secure Communication**: Ensure secure channels for distributing the public key and receiving encrypted updates.

## Performance Characteristics

- **Key Generation**: The most expensive operation, scales with key size (~1 second for 2048-bit keys on modern hardware)
- **Encryption**: Linear with the size of the vector (~0.02 seconds per value with 2048-bit keys)
- **Aggregation**: Very fast, almost negligible overhead
- **Decryption**: Linear with the size of the vector (~0.01 seconds per value with 2048-bit keys)

## Testing

Comprehensive unit tests are provided in `tests/smc/test_paillier.py`, which verify:

- Key generation functionality
- Encryption/decryption correctness
- Homomorphic properties
- Secure aggregation
- Error handling for edge cases

To run the tests:
```bash
cd /path/to/federated-learning-prototype
PYTHONPATH=. pytest -xvs tests/smc/test_paillier.py
```

## Demo

A demonstration script is available at `src/smc/demo_paillier.py`, which illustrates the entire workflow from key generation to secure aggregation and verification.

To run the demo:
```bash
cd /path/to/federated-learning-prototype
python -m src.smc.demo_paillier
```
