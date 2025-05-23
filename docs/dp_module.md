# Differential Privacy Module

This document describes the Differential Privacy (DP) module implemented in the federated learning prototype. The module provides mechanisms to add calibrated noise to model parameters to preserve privacy of the training data.

## Overview

Differential Privacy provides formal privacy guarantees by adding carefully calibrated noise to query outputs. In federated learning, DP can be applied at different stages:

1. **Local DP**: Each client adds noise to their local model updates before sharing with the server
2. **Global DP**: The server adds noise to the aggregated model after combining updates from clients

This implementation focuses on **Global DP** by providing a Gaussian noise mechanism that can be integrated with the aggregation pipeline.

## Gaussian Mechanism

The Gaussian mechanism is implemented in `GaussianMechanism` class, which adds Gaussian noise calibrated according to the privacy parameters:

- **Epsilon (ε)**: Controls the privacy budget; smaller values provide stronger privacy guarantees
- **Delta (δ)**: The probability of privacy breach; typically set to a small value (e.g., 1e-5)
- **Sensitivity**: The maximum influence one record can have on the output; determined by clipping

The noise scale (standard deviation σ) is calculated as:

```
σ = sqrt(2 * log(1.25/δ)) * sensitivity / ε
```

## Usage

### Basic Usage

```python
from src.dp import GaussianMechanism

# Initialize the mechanism
dp_mechanism = GaussianMechanism(
    epsilon=1.0,
    delta=1e-5,
    sensitivity=1.0
)

# Add noise to a value
noisy_value = dp_mechanism.add_noise(original_value)

# Add noise to model weights
noisy_weights = dp_mechanism.add_noise_to_weights(model_weights)
```

### Integration with Aggregator

```python
from src.dp import GaussianMechanism, compute_sensitivity_l2

# After aggregating updates from clients
aggregated_weights = average_client_updates(client_updates)

# Compute sensitivity based on clipping norm and number of clients
sensitivity = compute_sensitivity_l2(aggregated_weights, clip_norm=1.0) / num_clients

# Create DP mechanism
dp_mechanism = GaussianMechanism(
    epsilon=1.0,
    delta=1e-5,
    sensitivity=sensitivity
)

# Apply privacy
private_weights = dp_mechanism.add_noise_to_weights(aggregated_weights)

# Update global model with private weights
update_global_model(private_weights)
```

### Advanced Configuration

For different layers that might have different sensitivities:

```python
# Per-layer sensitivities
weight_sensitivity = {
    'layer1.weight': 1.0,
    'layer2.weight': 0.5,
    'output.weight': 0.1
}

# Apply noise with layer-specific sensitivities
noisy_weights = dp_mechanism.add_noise_to_weights(
    model_weights, 
    weight_sensitivity=weight_sensitivity
)
```

## Utility Functions

The module provides utility functions for sensitivity analysis and privacy accounting:

- `compute_sensitivity_l2()`: Compute the L2 sensitivity based on clipping norm
- `compute_privacy_spent()`: Calculate the privacy cost for multiple iterations
- `calibrate_noise_to_privacy()`: Determine noise scale for given privacy parameters
- `assess_utility_loss()`: Measure the utility impact of adding privacy noise

## Privacy-Utility Tradeoff

There is an inherent tradeoff between privacy (lower ε) and utility (model accuracy):

- Stronger privacy (lower ε): More noise, potentially lower model accuracy
- Weaker privacy (higher ε): Less noise, better model accuracy

The implementation allows experimentation with different privacy parameters to find an appropriate balance for specific use cases.

## Testing

The module includes comprehensive test coverage:

- Unit tests for `GaussianMechanism` class and utility functions
- Integration tests demonstrating usage in federated learning settings
- Tests for privacy-utility tradeoff analysis

Run the tests with:

```bash
pytest tests/dp/
```
