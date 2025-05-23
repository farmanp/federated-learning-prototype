# Aggregator Pipeline Implementation - Ticket #5

This document provides instructions for running and testing the Aggregator Pipeline implementation.

## Requirements

- Python 3.9 or higher
- Dependencies listed in `pyproject.toml`

## Setup

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

## Running the Aggregator Server

Run the aggregator server with:

```bash
python -m src.aggregator.main --config config/aggregator.yaml
```

The server will:
1. Load the configuration
2. Generate Paillier encryption keys
3. Initialize the Aggregator Pipeline with the specified parameters
4. Start the gRPC server to receive client connections

## Implementation Overview

### Core Components

1. **Aggregator Pipeline** (`src/aggregator/pipeline.py`)
   - Handles the main aggregation logic with homomorphic encryption and differential privacy
   - Validates client updates and enforces security requirements
   - Tracks privacy budget usage across multiple rounds

2. **gRPC Server** (`src/communication/grpc_server.py`)
   - Provides a secure communication interface for clients
   - Receives encrypted model updates and distributes the aggregated model
   - Handles client authentication and validation

3. **Integration Tests** (`tests/integration/test_aggregator_integration.py`)
   - Validates the complete federated learning workflow
   - Tests multiple client scenarios and privacy guarantees

### Configuration

The system is configured via YAML files in the `config/` directory:

- `aggregator.yaml`: Contains settings for the aggregator server, including:
  - Server address and port
  - Paillier key size
  - Differential privacy parameters
  - Logging configuration

## Testing

Run the tests with:

```bash
# Run unit tests for the aggregator pipeline
python -m pytest tests/aggregator/test_pipeline.py -v

# Run integration tests
python -m pytest tests/integration/test_aggregator_integration.py -v
```

## Next Steps

The implementation is complete and passes all tests. The following tasks remain:

1. Implement a corresponding client for the gRPC service
2. Conduct a full end-to-end system test with multiple data parties
3. Optimize performance for larger models
4. Conduct security audits and penetration testing

## Definition of Done

- [x] Implementation of AggregatorPipeline with SMC and DP integration
- [x] Comprehensive unit tests for all functionality
- [x] Documentation of the API and security features
- [x] gRPC server implementation for client communication
- [ ] Integration with a real federated learning scenario (pending client implementation)
- [x] Code review and approval

## Contributors

- Faisal Pirzada (Ticket #5 implementation)
