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

### Option 1: Using pip/venv (Standard)

1. Clone this repository
2. Set up a virtual environment and install dependencies:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

### Option 2: Using Docker (Recommended)

For easier setup and consistency across environments, use Docker:

1. Clone this repository
2. Build and run using Docker Compose:

```bash
# Build and start all services
docker-compose -f docker/docker-compose.dev.yml up --build

# Or build and start just the data party
docker-compose -f docker/docker-compose.dev.yml up --build data_party_dev
```

See [Docker Setup Guide](docs/docker_setup_guide.md) for more details.

## Project Structure

```
federated-learning-prototype/
├── src/
│   ├── aggregator/          # Central aggregation server
│   ├── data_party/          # Local data party implementations
│   ├── smc/                 # Secure Multiparty Computation
│       └── paillier.py      # Paillier homomorphic encryption
│   ├── dp/                  # Differential Privacy mechanisms
│   ├── models/              # ML model implementations
│       └── trainer.py       # Local logistic regression trainer
│   ├── communication/       # Inter-party communication
│   └── utils/               # Common utilities
│       └── data_loader.py   # Dataset loading and preprocessing
├── tests/                   # Test suite
├── docker/                  # Docker configurations
├── config/                  # Configuration files
├── docs/                    # Documentation
│   ├── implementation_guide.md     # Aggregator implementation guide
│   ├── data_party_guide.md         # Data party implementation guide
│   ├── docker_setup_guide.md       # Docker setup instructions
│   ├── data_module.md              # Data loading documentation
│   ├── dp_module.md                # Differential privacy documentation
│   └── paillier_module.md          # Secure multiparty computation docs
└── pyproject.toml          # Project dependencies
```

## Documentation

- [Aggregator Implementation Guide](docs/implementation_guide.md): Instructions for the central aggregation server
- [Data Party Guide](docs/data_party_guide.md): How to run data party nodes
- [Docker Setup Guide](docs/docker_setup_guide.md): Instructions for using Docker containers
- [Data Module Documentation](docs/data_module.md): Details about data loading and processing
- [Differential Privacy Module](docs/dp_module.md): Information about the DP mechanisms
- [Paillier Module Documentation](docs/paillier_module.md): Details about the SMC implementation

## Data Module

The project includes a data loading and preprocessing module (`src/utils/data_loader.py`) that provides essential functionality for the federated learning simulation:

### Features

- **Data Loading**: Load real data from CSV files or generate synthetic classification data
- **Preprocessing**: Normalize features, handle missing values, and split into train/test sets
- **Error Handling**: Robust error handling for various edge cases (missing files, empty datasets, etc.)

### Usage

```python
from src.utils.data_loader import load_data, preprocess_data

# Load data (either from CSV or synthetic)
df = load_data(file_path="path/to/data.csv")  # From CSV
# or
df = load_data(file_path=None, n_samples=1000, n_features=20)  # Synthetic

# Preprocess data
X_train, X_test, y_train, y_test = preprocess_data(
    df, 
    target_column='target',
    test_size=0.2
)
```

## Secure Multiparty Computation (SMC)

The project implements privacy-preserving computation using the Paillier homomorphic encryption scheme (`src/smc/paillier.py`):

### Features

- **Homomorphic Encryption**: Allows performing computations on encrypted data
- **Secure Aggregation**: Aggregate model updates without revealing individual contributions
- **Key Management**: Generate and manage cryptographic keys for secure operations

### Usage

```python
from src.smc.paillier import PaillierCrypto, aggregate_encrypted_vectors

# Initialize the cryptosystem
crypto = PaillierCrypto()
public_key, private_key = crypto.generate_keys()

# Encrypt model weights (at data parties)
encrypted_weights = crypto.encrypt_vector([0.1, 0.2, 0.3])

# Aggregate encrypted weights (at aggregator)
aggregated = aggregate_encrypted_vectors([encrypted_weights1, encrypted_weights2])

# Decrypt the aggregated result (at aggregator)
decrypted_result = crypto.decrypt_vector(aggregated)
```

## Model Trainer

The project includes a local model trainer module (`src/models/trainer.py`) that trains logistic regression models on each data party's private data:

### Features

- **Local Training**: Train logistic regression models using scikit-learn
- **Parameter Extraction**: Extract model coefficients for federated aggregation
- **Performance Evaluation**: Calculate accuracy on local test data
- **Edge Case Handling**: Handle constant labels, missing data, and other edge cases

### Usage

```python
from src.models.trainer import train_local_model

# Train a logistic regression model on local data
weights, accuracy, metrics = train_local_model(X_train, y_train, X_test, y_test)

# Use custom hyperparameters
custom_params = {'solver': 'liblinear', 'max_iter': 2000, 'C': 0.5}
weights, accuracy, metrics = train_local_model(
    X_train, y_train, X_test, y_test, 
    hyperparams=custom_params
)
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