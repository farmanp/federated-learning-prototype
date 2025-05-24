# Data Party Implementation Guide

This document provides instructions for running and testing the Data Party implementation, which represents an individual participant in the federated learning system.

## Requirements

- Python 3.9 or higher
- Dependencies listed in `pyproject.toml`
- Running Aggregator service (see [Implementation Guide](implementation_guide.md))

## Setup Options

### Option 1: Local Setup

1. Set up the Python virtual environment:

```bash
# Make the script executable if needed
chmod +x setup_venv.sh

# Run the setup script
./setup_venv.sh

# Activate the virtual environment
source venv/bin/activate
```

2. Generate gRPC code:

```bash
# Make the script executable if needed
chmod +x src/communication/generate_grpc.sh

# Run the script
./src/communication/generate_grpc.sh
```

### Option 2: Docker Setup (Recommended)

For easier deployment and consistency across environments, use the Docker setup:

1. Build the data party container:

```bash
docker-compose -f docker/docker-compose.dev.yml build --no-cache data_party_dev
```

2. Start the data party container:

```bash
docker-compose -f docker/docker-compose.dev.yml up -d data_party_dev
```

For more detailed Docker instructions, see the [Docker Setup Guide](docker_setup_guide.md).

## Running the Data Party

### Running Locally

Run the data party with:

```bash
python -m src.data_party.main --config config/party_1.yaml --party-id 1
```

### Running in Docker

If using Docker, run the data party with:

```bash
docker exec -it docker-data_party_dev-1 python -m src.data_party.main --config /app/config/party_1.yaml --party-id 1
```

You can substitute with different party configurations by changing the config file and party-id:

```bash
docker exec -it docker-data_party_dev-1 python -m src.data_party.main --config /app/config/party_2.yaml --party-id 2
```

## Data Preparation

Each data party needs a dataset to operate on. You can use the provided script to generate sample data for testing:

```bash
# Local execution
python create_party_data.py

# Docker execution
docker exec -it docker-data_party_dev-1 python /app/create_party_data.py
```

The script will create sample data files in the `data/party_[1-3]/` directories.

## Configuration

The data party is configured via YAML files in the `config/` directory:

- `party_1.yaml`, `party_2.yaml`, `party_3.yaml`: Configuration files for different parties with settings for:
  - Party ID, host, and port
  - Aggregator connection details
  - Dataset path and parameters
  - Training configuration
  - Model parameters
  - Security settings
  - Logging configuration

## Implementation Overview

### Core Components

1. **Data Party** (`src/data_party/main.py`)
   - Loads and preprocesses local data
   - Trains local models
   - Encrypts model updates
   - Communicates with the Aggregator

2. **gRPC Client** (`src/communication/grpc_client.py`)
   - Connects to the Aggregator's gRPC server
   - Sends encrypted model updates
   - Receives aggregated model updates

3. **Local Model Trainer** (`src/models/trainer.py`)
   - Implements local model training algorithms
   - Supports various model types (decision trees, logistic regression, etc.)

4. **Data Loader** (`src/utils/data_loader.py`)
   - Loads data from CSV files
   - Preprocesses data for training

## Testing

To test the data party implementation:

```bash
# Run unit tests for the data party
python -m pytest tests/data_party -v

# Run integration tests
python -m pytest tests/integration -v
```

## Troubleshooting

### Connection Issues

If the data party can't connect to the aggregator:
1. Ensure the aggregator is running
2. Check that the host and port in the config file are correct
3. Verify network connectivity between the data party and aggregator

### Data Loading Problems

If encountering data loading errors:
1. Verify the data file exists at the specified path
2. Check permissions on the data directory
3. Validate that the CSV file format matches expected schema

When using Docker:
1. Ensure the data directory is properly mounted
2. Check that the paths in the config files are correct relative to the container's filesystem

## Next Steps

After running the data parties successfully:
1. Experiment with different model parameters
2. Test with larger or different datasets
3. Implement additional model types
4. Optimize local training for better performance
