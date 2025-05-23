# Federated Learning Prototype with SMC and Differential Privacy

A hybrid approach to privacy-preserving federated learning that integrates Secure Multiparty Computation (SMC) and Differential Privacy (DP) to enhance privacy without compromising model accuracy.

## Overview

This prototype implements the novel federated learning framework described in "A Hybrid Approach to Privacy-Preserving Federated Learning" paper. The system addresses the limitations of using SMC or DP in isolation, offering a scalable solution that mitigates inference attacks during training and from the final model.

## System Architecture

### Components

- **Data Parties**: Entities holding local datasets and performing local model training
- **Aggregator**: Central entity that aggregates encrypted model updates
- **SMC Protocol**: Ensures secure aggregation without revealing individual updates (Threshold Paillier Cryptosystem)
- **DP Mechanism**: Adds noise to the aggregated model to prevent inference attacks (Laplace/Gaussian mechanisms)

### Data Flow

1. Each data party trains a local model and encrypts the updates using SMC
2. Encrypted updates are sent to the aggregator
3. Aggregator performs secure aggregation and adds DP noise
4. Aggregated model is decrypted and shared with all parties

## Supported Models

- Decision Trees
- Convolutional Neural Networks (CNNs)
- Linear Support Vector Machines (SVMs)

## Requirements

- Python 3.8+
- uv package manager

## Installation

1. Clone this repository
2. Install dependencies using uv:

```bash
uv sync
```

For development dependencies:

```bash
uv sync --group dev
```

## Project Structure

```
federated-learning-prototype/
├── src/
│   ├── aggregator/          # Central aggregation server
│   ├── data_party/          # Local data party implementations
│   ├── smc/                 # Secure Multiparty Computation
│   ├── dp/                  # Differential Privacy mechanisms
│   ├── models/              # ML model implementations
│   ├── communication/       # Inter-party communication
│   └── utils/               # Common utilities
├── tests/                   # Test suite
├── docker/                  # Docker configurations
├── config/                  # Configuration files
└── pyproject.toml          # Project dependencies
```

## Usage

### Running the Aggregator

```bash
uv run fl-aggregator --config config/aggregator.yaml
```

### Running a Data Party

```bash
uv run fl-data-party --party-id 1 --config config/party1.yaml
```

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black src/
uv run flake8 src/
```

### Type Checking

```bash
uv run mypy src/
```

## Docker Support

Docker containers are provided to simulate multiple data parties:

```bash
docker-compose up --scale data-party=5
```

## Performance Characteristics

- Scalable to a large number of participants
- Maintains high model accuracy despite added noise
- Manageable overhead from encryption and noise addition

## License

MIT License

## Contributing

Please read the contributing guidelines before submitting pull requests.