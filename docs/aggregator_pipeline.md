# Aggregator Pipeline

This document describes the Aggregator Pipeline implemented in the federated learning prototype. The pipeline combines secure multi-party computation with differential privacy to enable privacy-preserving federated learning.

## Overview

The Aggregator Pipeline provides a comprehensive server-side solution for securely aggregating model updates from multiple clients while preserving privacy. It integrates:

1. **Paillier Homomorphic Encryption**: For secure aggregation of encrypted updates
2. **Differential Privacy**: For adding calibrated noise to prevent inference attacks 
3. **gRPC Communication**: For secure, efficient client-server communication

## Pipeline Components

### 1. Encrypted Updates Collection

The pipeline accepts encrypted model updates from multiple clients in the form:
- `List[List[EncryptedNumber]]`: A list of encrypted weight vectors, one per client

### 2. Secure Aggregation

Using Paillier homomorphic encryption, the pipeline can:
- Validate the structure and consistency of encrypted updates
- Aggregate them securely without decrypting individual contributions
- This preserves the privacy of individual clients' updates

### 3. Differential Privacy Protection

After aggregation and decryption, the pipeline:
- Adds calibrated Gaussian noise to the aggregated model
- Scales noise based on privacy parameters (epsilon, delta, sensitivity)
- Provides formal privacy guarantees against inference attacks

### 4. Results Finalization

The final output is:
- A decrypted, privacy-protected weight vector: `List[float]`
- Privacy accounting information for tracking budget usage

## Usage

### Basic Usage

```python
from src.aggregator.pipeline import AggregatorPipeline
from src.smc.paillier import generate_keys

# Generate encryption keys
public_key, private_key = generate_keys()

# Initialize pipeline
pipeline = AggregatorPipeline(
    public_key=public_key, 
    private_key=private_key,
    dp_epsilon=1.0,
    dp_delta=1e-5,
    dp_sensitivity=1.0
)

# Process encrypted updates
final_weights = pipeline.aggregate_and_finalize(encrypted_updates)
```

### Simplified Interface

For one-shot usage, a convenience function is provided:

```python
from src.aggregator.pipeline import aggregate_and_finalize

final_weights = aggregate_and_finalize(
    encrypted_updates=encrypted_updates,
    public_key=public_key,
    private_key=private_key,
    dp_params={'epsilon': 1.0, 'delta': 1e-5, 'sensitivity': 1.0}
)
```

## Privacy Parameters

The differential privacy protection can be customized with:

- **Epsilon (ε)**: The privacy budget; smaller values provide stronger privacy
- **Delta (δ)**: The probability of privacy breach; typically set to a small value (e.g., 1e-5)
- **Sensitivity**: The maximum influence one record can have on the output

These parameters can be adjusted for each aggregation round:

```python
# Create custom DP parameters for this round
dp_params = {
    'epsilon': 0.5,    # Stronger privacy for this round
    'delta': 1e-6,     # Lower failure probability
    'sensitivity': 0.5  # Lower sensitivity for this round
}

# Use custom parameters for this round
result = pipeline.aggregate_and_finalize(encrypted_updates, dp_params=dp_params)
```

## Privacy Budget Tracking

The pipeline keeps track of privacy budget usage across multiple rounds:

```python
# Get current privacy budget usage
budget_info = pipeline.get_privacy_budget_spent()
print(f"Effective epsilon after {budget_info['rounds_completed']} rounds: {budget_info['effective_epsilon']}")
```

## Metrics and Monitoring

The pipeline logs detailed metrics about each aggregation round:
- Client count and vector dimensions
- Statistical properties of aggregated weights
- Noise impact on model weights
- Current privacy parameters

These metrics help monitor the privacy-utility tradeoff over time.

## gRPC Integration

The Aggregator Pipeline is integrated with a gRPC server for secure, efficient communication with clients:

### Server Implementation

The server exposes the following gRPC services:

1. **GetPublicKey**: Provides clients with the Paillier public key needed for encryption
2. **SubmitUpdate**: Receives encrypted model updates from clients
3. **GetAggregatedModel**: Returns the aggregated model after processing client updates

### Starting the Server

```python
from src.smc.paillier import generate_keys
from src.aggregator.pipeline import AggregatorPipeline
from src.communication.grpc_server import serve

# Generate keys and create aggregator pipeline
public_key, private_key = generate_keys()
aggregator = AggregatorPipeline(
    public_key=public_key,
    private_key=private_key,
    dp_epsilon=1.0,
    dp_delta=1e-5,
    dp_sensitivity=1.0
)

# Start the gRPC server
serve(
    host="localhost",
    port=50051,
    public_key=public_key,
    private_key=private_key,
    aggregator_pipeline=aggregator
)
```

### Client-Server Flow

1. Client requests the public key from the aggregator
2. Client trains a local model and encrypts the updates
3. Client sends encrypted updates to the aggregator
4. Aggregator validates and processes the updates
5. When enough clients have contributed, aggregation is performed
6. Clients can request the final aggregated model

This distributed architecture enables secure, privacy-preserving federated learning across multiple data parties.
