"""
Main entry point for the Aggregator service.
"""
from pathlib import Path
import click
from loguru import logger
import yaml

from src.smc.paillier import generate_keys
from src.aggregator.pipeline import AggregatorPipeline
from src.communication.grpc_server import serve


@click.command()
@click.option('--config', 'config_path', default='config/aggregator.yaml', help='Path to the aggregator configuration file.')
def main(config_path: str):
    """
    Starts the Aggregator service.
    """
    logger.info(f"Initializing Aggregator with config: {config_path}")

    # Load configuration
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return

    try:
        with open(config_file, 'r', encoding='utf-8') as f_config:
            config = yaml.safe_load(f_config)
    except (yaml.YAMLError, IOError) as e_yaml:
        logger.error(f"Error loading configuration: {e_yaml}")
        return

    # Initialize SMC components (Paillier keys)
    logger.info("Initializing secure multi-party computation components...")
    paillier_config = config.get('security', {}).get('paillier', {})
    key_size = paillier_config.get('key_size', 2048)

    try:
        public_key, private_key = generate_keys(key_size=key_size)
        logger.success(f"Generated Paillier keys with size {key_size}")
    except (ValueError, RuntimeError) as e_keys:
        logger.error(f"Error generating Paillier keys: {e_keys}")
        return

    # Extract DP configuration
    dp_config = config.get('differential_privacy', {})
    dp_enabled = dp_config.get('enabled', False)

    if dp_enabled:
        epsilon = float(dp_config.get('epsilon', 1.0))
        delta = float(dp_config.get('delta', 1e-5))  # Ensure delta is float
        clip_norm = float(dp_config.get('clip_norm', 1.0))

        logger.info(f"Differential Privacy enabled: epsilon={epsilon}, delta={delta}, clip_norm={clip_norm}")

        # Initialize Aggregator Pipeline with DP
        aggregator = AggregatorPipeline(
            public_key=public_key,
            private_key=private_key,
            dp_epsilon=epsilon,
            dp_delta=delta,
            dp_sensitivity=clip_norm
        )
    else:
        logger.info("Differential Privacy is disabled")

        # Initialize Aggregator Pipeline with minimal DP (very high epsilon)
        aggregator = AggregatorPipeline(
            public_key=public_key,
            private_key=private_key,
            # Ensure these are float for consistency, though defaults are already
            dp_epsilon=float(1000.0),
            dp_delta=float(1e-9),
            dp_sensitivity=float(0.1)
        )

    # Start gRPC server to listen for Data Parties
    server_config = config.get('aggregator', {})
    host = server_config.get('host', '0.0.0.0')  # Changed from 'localhost'
    port = server_config.get('port', 50051)
    max_workers = server_config.get('max_workers', 10)

    logger.info(f"Starting gRPC server on {host}:{port} with {max_workers} workers...")

    try:
        # This will start the server in the current thread
        serve(
            host=host,
            port=port,
            public_key=public_key,
            private_key=private_key,
            aggregator_pipeline=aggregator,
            max_workers=max_workers
        )
    except (OSError, RuntimeError) as e_server:
        logger.error(f"Error starting gRPC server: {e_server}")
        return

    # Note: The serve function has an infinite loop, so the code below won't be reached
    # unless the server is explicitly stopped or an unhandled exception occurs.
    logger.info("Aggregator service shutting down.")


if __name__ == '__main__':
    main()
