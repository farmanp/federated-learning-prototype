"""
Main entry point for the Aggregator service.
"""
import click
from loguru import logger
import yaml
from pathlib import Path

from src.dp.aggregator import DPAggregator


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
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    # Extract DP configuration
    dp_config = config.get('differential_privacy', {})
    dp_enabled = dp_config.get('enabled', False)
    
    if dp_enabled:
        epsilon = dp_config.get('epsilon', 1.0)
        delta = dp_config.get('delta', 1e-5)
        clip_norm = dp_config.get('clip_norm', 1.0)
        
        logger.info(f"Differential Privacy enabled: epsilon={epsilon}, delta={delta}, clip_norm={clip_norm}")
        
        # Initialize DP Aggregator
        dp_aggregator = DPAggregator(
            epsilon=epsilon,
            delta=delta,
            clip_norm=clip_norm
        )
    else:
        logger.info("Differential Privacy is disabled")
        dp_aggregator = None
    
    # TODO: Initialize SMC components (e.g., Paillier keys)
    # TODO: Start gRPC server to listen for Data Parties
    # TODO: Implement aggregation logic (using dp_aggregator if enabled)
    
    logger.info("Aggregator service started successfully.")


if __name__ == '__main__':
    main()
