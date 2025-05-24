# Docker Setup Guide for Federated Learning Prototype

This guide provides instructions for running the federated learning prototype components using Docker containers for simplified deployment and better isolation.

## Prerequisites

- Docker Engine (v20.10+)
- Docker Compose (v2.0+)
- Git (to clone the repository if you haven't already)

## Docker Components

The federated learning prototype includes Docker support for running:

1. **Aggregator Service**: The central node that aggregates encrypted model updates
2. **Data Party Services**: Multiple data parties that train local models on their datasets

## Quick Start with Docker Compose

The easiest way to run the entire system is using Docker Compose:

```bash
cd federated-learning-prototype
docker-compose -f docker/docker-compose.dev.yml up
```

This starts both the aggregator and a data party container in development mode with mounted source code for easier testing and debugging.

## Building Individual Containers

If you need to rebuild the containers with the latest code changes:

```bash
# For the aggregator
docker-compose -f docker/docker-compose.dev.yml build aggregator

# For the data party
docker-compose -f docker/docker-compose.dev.yml build --no-cache data_party_dev
```

## Development Mode

Development mode mounts the source code from your host machine into the containers, allowing for faster iteration without rebuilding the containers for every code change:

```bash
docker-compose -f docker/docker-compose.dev.yml up data_party_dev
```

### Running Data Party with Configuration

To execute a data party with a specific configuration file:

```bash
docker exec -it docker-data_party_dev-1 python -m src.data_party.main --config /app/config/party_1.yaml --party-id 1
```

You can substitute `party_1.yaml` with `party_2.yaml` or `party_3.yaml` to use different configurations.

## Configuration Files

The container mounts the `config/` directory, allowing you to edit configuration files on your host machine and have those changes reflected immediately in the container:

- `config/aggregator.yaml`: Configuration for the aggregator service
- `config/party_*.yaml`: Configurations for different data parties

## Data Management

The container also mounts the `data/` directory. The expected structure is:

```
data/
├── party_1/
│   └── iris_party_1.csv
├── party_2/
│   └── iris_party_2.csv
└── party_3/
    └── iris_party_3.csv
```

You can generate sample data for all parties using:

```bash
python create_party_data.py
```

## Troubleshooting Docker Setup

### Unable to Connect to Aggregator

If the data party cannot connect to the aggregator, verify that:

1. Both containers are running (`docker ps`)
2. The aggregator service is started before the data party attempts to connect
3. The configuration files use the correct hostname (`aggregator`) for the aggregator service

### Missing Dependencies

If you encounter missing dependencies, you may need to rebuild the container:

```bash
docker-compose -f docker/docker-compose.dev.yml build --no-cache data_party_dev
```

### Persistent Volume Issues

If changes to dependencies aren't reflected after rebuilding:

1. Stop all containers: `docker-compose -f docker/docker-compose.dev.yml down`
2. Remove any persistent volumes: `docker volume prune` (use caution as this removes all unused volumes)
3. Rebuild and start: `docker-compose -f docker/docker-compose.dev.yml up --build`

## Next Steps

After becoming familiar with the Docker setup, you can:

1. Implement your own federated learning algorithms
2. Modify configuration parameters to test different privacy settings
3. Add your own datasets to test performance on different problems
