# Quick Start Guide: Running Data Party Services

This guide provides a quick way to get started with the data party services in the federated learning prototype.

## Prerequisites

- Docker and Docker Compose installed
- Git repository cloned

## 1. Build and Start the Container

```bash
# Navigate to the project directory
cd federated-learning-prototype

# Build and start the data party container in development mode
docker-compose -f docker/docker-compose.dev.yml up -d data_party_dev
```

## 2. Generate Sample Data

```bash
# Create sample data for all parties
python create_party_data.py
```

This will create data files in the following structure:
```
data/
├── party_1/iris_party_1.csv
├── party_2/iris_party_2.csv
└── party_3/iris_party_3.csv
```

## 3. Run Data Party Services

You can run multiple data party instances with different configurations:

### Party 1
```bash
docker exec -it docker-data_party_dev-1 python -m src.data_party.main --config /app/config/party_1.yaml --party-id 1
```

### Party 2
```bash
docker exec -it docker-data_party_dev-1 python -m src.data_party.main --config /app/config/party_2.yaml --party-id 2
```

### Party 3
```bash
docker exec -it docker-data_party_dev-1 python -m src.data_party.main --config /app/config/party_3.yaml --party-id 3
```

## 4. Test Connection to Aggregator

To fully test the federated learning system, the aggregator service must be running. Start the aggregator in a separate terminal:

```bash
docker-compose -f docker/docker-compose.dev.yml up -d aggregator
```

Once the aggregator is running, the data parties will be able to connect, submit model updates, and receive aggregated models.

## 5. View Logs

To view logs from the data party container:

```bash
# View live logs
docker logs -f docker-data_party_dev-1

# View recent logs
docker logs docker-data_party_dev-1 --tail 100
```

## 6. Stop the Services

When you're done:

```bash
# Stop all services
docker-compose -f docker/docker-compose.dev.yml down

# Or stop just the data party
docker-compose -f docker/docker-compose.dev.yml stop data_party_dev
```

## Next Steps

- Read the [Data Party Guide](data_party_guide.md) for more details on the implementation
- Explore [Docker Setup Guide](docker_setup_guide.md) for advanced Docker configuration
- Check the [Implementation Guide](implementation_guide.md) for details on the aggregator service

## Troubleshooting

### Container Won't Start
If the container fails to start, check:
```bash
docker logs docker-data_party_dev-1
```

### Dependencies Missing
If dependencies are missing, rebuild the container:
```bash
docker-compose -f docker/docker-compose.dev.yml build --no-cache data_party_dev
```

### Cannot Connect to Aggregator
Make sure the aggregator service is running and properly configured in the party's YAML file.
