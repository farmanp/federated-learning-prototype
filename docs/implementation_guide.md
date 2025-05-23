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

## gRPC Code Generation and Docker Setup

This section details the process for generating gRPC Python client/server stubs from `.proto` definitions and how this integrates with the Docker setup, particularly for the development environment.

### Overview

The project uses gRPC for communication between services (e.g., Aggregator and Data Parties). Python code for gRPC is generated from `.proto` files. Docker is used to containerize the services. A key challenge is ensuring that the gRPC code is correctly generated and available to the Python runtime, especially when using Docker volumes for development, which can overwrite files generated during the image build.

### Key Files

-   `src/communication/proto/*.proto`: Contains the Protocol Buffer definitions for services and messages.
-   `src/communication/generate_grpc.sh`: A shell script responsible for invoking the gRPC tools to generate Python code.
-   `docker/Dockerfile.aggregator`: The Dockerfile for building the aggregator service image.
-   `docker/docker-compose.dev.yml`: The Docker Compose file used for local development, which includes volume mounts for live code reloading.

### gRPC Code Generation (`src/communication/generate_grpc.sh`)

This script automates the generation of Python gRPC files. Its main responsibilities are:

1.  **Environment Check**: It verifies that `python -m grpc_tools.protoc` is accessible. `grpcio-tools` (which provides this module) should be listed as a dependency in `pyproject.toml` and installed via `uv sync`.
2.  **File Discovery**: It finds all `.proto` files within the `src/communication/proto/` directory.
3.  **Code Generation**: It invokes `python -m grpc_tools.protoc` with appropriate flags:
    *   `--proto_path`: Specifies the directory containing the `.proto` files.
    *   `--python_out`: Specifies the output directory for the `*_pb2.py` (message serialization) files.
    *   `--grpc_python_out`: Specifies the output directory for the `*_pb2_grpc.py` (client/server stub) files.
    Both output directories are set to `src/communication/generated/`.
4.  **Import Path Correction (Crucial Step)**: After generation, the script uses `sed` to modify the `*_pb2_grpc.py` files. It changes absolute imports (e.g., `import aggregator_service_pb2`) to relative imports (e.g., `from . import aggregator_service_pb2`). This is essential because the generated files reside within a Python package (`src.communication.generated`), and relative imports ensure that Python can correctly locate the sibling `*_pb2.py` modules.

### Docker Integration (`Dockerfile.aggregator`)

The Dockerfile for the aggregator service includes steps to generate the gRPC code during the image build process:

1.  **Copy Files**: Project files, including `pyproject.toml`, `uv.lock`, and the `src/` directory (containing the `.proto` files and `generate_grpc.sh`), are copied into the image.
2.  **Install Dependencies**: `RUN uv sync --frozen` installs all Python dependencies, including `grpcio-tools`.
3.  **Set Permissions**: `RUN chmod +x src/communication/generate_grpc.sh` makes the generation script executable.
4.  **Generate Code**: `RUN uv run src/communication/generate_grpc.sh` executes the script. Using `uv run` ensures that the script runs within the Python environment managed by `uv`, where `grpcio-tools` is available. This step creates the `src/communication/generated/` directory and populates it with the necessary Python files *within the Docker image*.

### Development Environment (`docker-compose.dev.yml`)

The development Docker Compose setup presents a specific challenge due to volume mounting:

-   **Volume Mounting**: The line `volumes: - ../src:/app/src` in `docker-compose.dev.yml` mounts your local `src` directory into the `/app/src` directory inside the container. This is great for live code reloading but means that the `src/communication/generated/` directory created during the `docker build` phase (which is part of the image layer) gets obscured or overwritten by the content of your local `src/communication/generated/` directory (which might be empty or outdated).

-   **Solution**: To address this, the `command` for the `aggregator` service in `docker-compose.dev.yml` is crafted to re-run the `generate_grpc.sh` script *every time the container starts*:
    ```yaml
    services:
      aggregator:
        # ... other configurations ...
        command: ["sh", "-c", "cd /app/src/communication && uv run ./generate_grpc.sh && echo 'Contents of /app/src/communication/generated after script:' && ls -la /app/src/communication/generated && cd /app && uv run python -m src.aggregator.main"]
        # ... other configurations ...
    ```
    This command sequence does the following upon container startup:
    1.  Changes to the `/app/src/communication` directory.
    2.  Executes `uv run ./generate_grpc.sh`, regenerating the gRPC files directly into the mounted volume space.
    3.  Changes back to the `/app` directory.
    4.  Starts the main aggregator application.
    This ensures that the generated gRPC files are always present and up-to-date in the environment Python sees at runtime, reflecting any changes to `.proto` files after a simple container restart (`docker-compose restart aggregator` or `docker-compose up -d --build aggregator`).

### Troubleshooting Common Issues

-   **`ModuleNotFoundError` for `*_pb2.py` or `*_pb2_grpc.py` files**:
    *   Verify that the `generate_grpc.sh` script ran successfully. Check container logs.
    *   Ensure the import correction step (absolute to relative) in `generate_grpc.sh` is working. The `*_pb2_grpc.py` file should use `from . import ...`.
    *   Confirm that `src/communication/generated/__init__.py` exists, making it a package.
    *   If using Docker Compose with volumes, ensure the script is run at container startup as described above.

-   **`grpc_tools.protoc` not found or script errors during generation**:
    *   Make sure `grpcio-tools` is listed in your `pyproject.toml`.
    *   Ensure `uv sync --frozen` (or equivalent `uv pip install -r requirements.txt`) is run before the script attempts generation, both in the Dockerfile and if running locally.
    *   Check execute permissions on `generate_grpc.sh` (`chmod +x`).
    *   Verify paths within the script and in the Docker/Compose commands are correct.

-   **TypeErrors for configuration values (e.g., `delta` for Differential Privacy)**:
    *   When loading numeric values (especially those in scientific notation like `1e-5`) from YAML files, ensure they are explicitly cast to the correct Python type (e.g., `float()`) in the application code that consumes them. `yaml.safe_load` might sometimes interpret such values as strings if not handled carefully downstream.


