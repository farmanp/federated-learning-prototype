"""
Main entry point for the Data Party service.
"""
import click
import yaml
import grpc  # Import grpc for RpcError handling
from loguru import logger
import numpy as np # For handling model weights

from src.communication.grpc_client import AggregatorClient
from src.communication.generated import aggregator_service_pb2 # For EncryptedValue
from src.utils.data_loader import load_data, preprocess_data
from src.models.trainer import train_local_model
from src.smc.paillier import PaillierCrypto, PublicKey # For Paillier operations


@click.command()
@click.option(
    '--config',
    'config_path',
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help='Path to the data party configuration file.'
)
@click.option(
    '--party-id',
    'party_id',
    required=True,
    help='Unique identifier for this data party.'
)
def main(config_path: str, party_id: str):
    """
    Starts a Data Party instance, loads configuration, and interacts with the Aggregator.
    """
    logger.info(f"Initializing Data Party '{party_id}' with config: {config_path}")

    # Load configuration
    try:
        with open(config_path, 'r', encoding='utf-8') as f_config:
            config = yaml.safe_load(f_config)
        logger.info("Configuration loaded successfully.")
        logger.debug(f"Config content: {config}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return
    except yaml.YAMLError as e_yaml:
        logger.error(f"Error parsing configuration file '{config_path}': {e_yaml}")
        return
    except IOError as e_io:
        logger.error(f"IOError reading configuration file '{config_path}': {e_io}")
        return
    except Exception as e_config_load:
        logger.error(
            f"Unexpected error loading configuration from '{config_path}': {e_config_load}"
        )
        return

    # Extract configurations
    aggregator_config = config.get('aggregator', {})
    aggregator_host = aggregator_config.get('host', 'localhost')
    aggregator_port = int(aggregator_config.get('port', 50051))

    dataset_config = config.get('dataset', {})
    dataset_path = dataset_config.get('path') # Can be None for synthetic data
    dataset_name = dataset_config.get('name', 'synthetic')
    
    training_config = config.get('training', {})
    local_epochs = training_config.get('epochs', 5) # Example, not directly used by current train_local_model
    
    # Initialize Paillier Crypto
    paillier_crypto = PaillierCrypto()
    aggregator_public_key: PublicKey | None = None

    # --- Load and Preprocess Data ---
    try:
        logger.info(f"Loading dataset: {dataset_name} (Path: {dataset_path or 'Synthetic'})")
        # Using party_id as part of random_state for synthetic data if multiple parties run locally
        # This ensures different synthetic datasets if party_id is e.g. an integer
        try:
            synthetic_random_state = int(party_id) if party_id.isdigit() else None
        except ValueError:
            synthetic_random_state = None
            
        df = load_data(file_path=dataset_path, random_state=synthetic_random_state if not dataset_path else dataset_config.get('random_seed'))
        
        X_train, X_test, y_train, y_test = preprocess_data(
            df,
            target_column=dataset_config.get('target_column', 'target'),
            test_size=dataset_config.get('test_size', 0.2),
            random_state=dataset_config.get('random_seed', 42)
        )
        logger.success("Data loading and preprocessing complete.")
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    except Exception as e_data:
        logger.error(f"Error during data loading or preprocessing: {e_data}")
        return

    logger.info(
        f"Attempting to connect to Aggregator at {aggregator_host}:{aggregator_port}"
    )
    client = None
    try:
        client = AggregatorClient(host=aggregator_host, port=aggregator_port)
        client.connect() # Added this line to explicitly call connect
        
        if not client.stub: # Check if connection was successful
            logger.error("Failed to connect to the aggregator. Aborting operations.")
            return

        logger.info("Successfully established connection with Aggregator.")

        # --- Federated Learning Loop (Simplified: one round) ---
        current_round_id = f"round_{party_id}_1"  # Example round ID

        # 1. Get public key from aggregator
        logger.debug(f"Requesting public key for round '{current_round_id}'...")
        public_key_response = client.get_public_key()
        if public_key_response and public_key_response.success and public_key_response.n and public_key_response.g:
            logger.info(
                f"Received public key (n_len={len(public_key_response.n)}, "
                f"g_len={len(public_key_response.g)})" # Log length of g as well
            )
            try:
                # Reconstruct Paillier PublicKey object
                # n and g are expected to be string representations of integers
                n_val = int(public_key_response.n)
                g_val = int(public_key_response.g)
                aggregator_public_key = PublicKey(n=n_val)
                # The phe library's PublicKey g is n_val + 1 by default if not specified.
                # If aggregator sends a specific g, it should be used.
                # For now, we assume g is implicitly handled or set if needed.
                # If aggregator_service.proto implies g is also part of the key, ensure it's used.
                # The phe.paillier.PaillierPublicKey constructor only takes n.
                # The g value is typically n+1 or a specific generator.
                # If the g from proto is essential and different, PaillierCrypto might need adjustment
                # or we assume the default g for the given n is sufficient.
                # For now, we'll just set the public key in our PaillierCrypto instance
                paillier_crypto.set_keys(public_key=aggregator_public_key)
                logger.success("Paillier public key reconstructed and set.")
            except ValueError as e_pk_reconstruct:
                logger.error(f"Failed to reconstruct public key from aggregator response: {e_pk_reconstruct}")
                aggregator_public_key = None # Ensure it's None if reconstruction fails
            except Exception as e_pk_set:
                logger.error(f"Error setting reconstructed public key: {e_pk_set}")
                aggregator_public_key = None
        else:
            err_msg = (
                public_key_response.error_message
                if public_key_response
                else "No response or success=false or key components missing"
            )
            logger.warning(f"Failed to get public key. Error: {err_msg}")
            # Decide if to proceed: For now, we'll proceed but encryption will be skipped.

        # 2. Train local model
        logger.info(f"Starting local model training for {local_epochs} epochs (hyperparams from training_config)...")
        model_weights, accuracy, metrics = train_local_model(
            X_train, y_train, X_test, y_test, 
            hyperparams=training_config.get('hyperparameters') # Pass hyperparams from config
        )
        logger.success(f"Local model training completed. Test Accuracy: {accuracy:.4f}")
        logger.debug(f"Raw model weights (first 5): {model_weights[:5]}")

        # 3. Prepare and Encrypt Model Updates
        encrypted_model_weights_proto = []
        if aggregator_public_key and paillier_crypto.public_key:
            logger.info("Encrypting model weights using Paillier public key...")
            try:
                # Ensure weights are floats for encryption
                float_weights = [float(w) for w in model_weights]
                
                # Paillier library encrypts numbers. Serialization to string for gRPC is needed.
                # This is a placeholder for actual encryption and serialization.
                # Each EncryptedNumber would need its parts (e.g., ciphertext, exponent)
                # converted to strings to fit the EncryptedValue proto.
                
                # Placeholder: Convert weights to string and put in EncryptedValue
                # This does NOT represent actual Paillier encryption.
                for weight_idx, weight in enumerate(float_weights):
                    # Actual encryption:
                    # encrypted_weight_obj = paillier_crypto.encrypt_value(weight)
                    # serialized_val, serialized_exp = serialize_encrypted_number(encrypted_weight_obj)
                    # encrypted_model_weights_proto.append(
                    #     aggregator_service_pb2.EncryptedValue(
                    #         value=serialized_val, exponent=serialized_exp
                    #     )
                    # )
                    
                    # --- Placeholder for serialization ---
                    # Simulating that we have a string representation of the encrypted value
                    # In a real scenario, this would be the result of paillier_crypto.encrypt_value()
                    # and then serializing that EncryptedNumber object.
                    # For the demo, we'll just send the stringified float.
                    # This part needs to be replaced with actual Paillier encryption and serialization.
                    placeholder_encrypted_value_str = str(weight) # NOT ENCRYPTED
                    placeholder_exponent_str = "0" # Placeholder
                    
                    encrypted_model_weights_proto.append(
                        aggregator_service_pb2.EncryptedValue(
                            value=placeholder_encrypted_value_str, # This should be serialized ciphertext
                            exponent=placeholder_exponent_str     # This should be serialized exponent
                        )
                    )
                logger.success(f"Model weights prepared for submission (placeholder encryption). Count: {len(encrypted_model_weights_proto)}")
            except Exception as e_encrypt:
                logger.error(f"Error during (placeholder) encryption of model weights: {e_encrypt}")
                encrypted_model_weights_proto = [] # Clear if error
        else:
            logger.warning("Aggregator public key not available. Skipping encryption. Sending raw weights (as strings).")
            # Fallback: send raw weights as strings if no public key (for testing/demo)
            for weight in model_weights:
                encrypted_model_weights_proto.append(
                    aggregator_service_pb2.EncryptedValue(value=str(weight), exponent="raw")
                )


        # 4. Submit model updates
        logger.debug(
            f"Submitting model update for party '{party_id}', "
            f"round '{current_round_id}'..."
        )
        submit_response = client.submit_update(
            client_id=party_id,
            encrypted_weights=encrypted_model_weights_proto, # Use the prepared list
            round_id=current_round_id
        )
        if submit_response and submit_response.success:
            logger.info(
                f"Model update submitted successfully: {submit_response.message}"
            )
        else:
            msg = (
                submit_response.message
                if submit_response
                else "No response or success=false"
            )
            logger.warning(f"Failed to submit model update. Message: {msg}")

        # 5. Get aggregated model (renamed from #4 to #5)
        logger.debug(
            f"Requesting aggregated model for round '{current_round_id}'..."
        )
        aggregated_model_response = client.get_aggregated_model(
            round_id=current_round_id
        )
        if aggregated_model_response and aggregated_model_response.success:
            aggr_msg = aggregated_model_response.message or "successful"
            logger.info(
                f"Received aggregated model for round "
                f"'{aggregated_model_response.round_id}'. Status: {aggr_msg}. "
                f"Weights count: {len(aggregated_model_response.aggregated_weights)}. "
                f"Epsilon spent: {aggregated_model_response.epsilon_spent}"
            )
        else:
            msg = (
                aggregated_model_response.message
                if aggregated_model_response
                else "No response or success=false"
            )
            logger.warning(f"Failed to get aggregated model. Message: {msg}")

    except grpc.RpcError as e_rpc:
        logger.error(
            f"gRPC communication error with Aggregator: "
            f"Code: {e_rpc.code()} Details: {e_rpc.details()}"
        )
    except Exception as e_comm: # Catch other unexpected errors during communication
        logger.error(
            f"An unexpected error occurred during Aggregator communication: {e_comm}"
        )
    finally:
        if client:
            logger.info("Closing connection to Aggregator.")
            client.close()

    # TODO: Implement full local training loop (multiple rounds)
    # TODO: Implement actual Paillier encryption and serialization/deserialization for EncryptedValue
    # TODO: Use aggregated model for further local training or evaluation
    logger.info(
        f"Data Party '{party_id}' operations completed (using placeholders for encryption)."
    )


if __name__ == '__main__':
    # This allows running the script directly.
    # Click handles parsing command-line arguments.
    # Example:
    # poetry run python src/data_party/main.py \\
    #   --config config/party_template.yaml \\
    #   --party-id party1
    main()
