"""
gRPC server implementation for the Aggregator service.
"""
import base64
import concurrent.futures
import json
import time
from typing import Dict, List, Any, Optional
import grpc
from loguru import logger

# Import the generated gRPC code
# Note: These imports will work after running generate_grpc.sh
# pylint: disable=import-error
from src.communication.generated import aggregator_service_pb2 as pb2
from src.communication.generated import aggregator_service_pb2_grpc as pb2_grpc
# pylint: enable=import-error

from src.smc.paillier import PublicKey, PrivateKey, EncryptedNumber


class AggregatorServicer(pb2_grpc.AggregatorServiceServicer):
    """
    Implementation of the Aggregator gRPC service.
    """
    
    def __init__(self, public_key: PublicKey, private_key: PrivateKey, aggregator_pipeline):
        """
        Initialize the Aggregator service.
        
        Args:
            public_key: Paillier public key.
            private_key: Paillier private key.
            aggregator_pipeline: Instance of AggregatorPipeline.
        """
        self.public_key = public_key
        self.private_key = private_key
        self.aggregator_pipeline = aggregator_pipeline
        
        # Storage for client updates
        self.client_updates: Dict[str, Dict[str, List[EncryptedNumber]]] = {}
        self.aggregation_results: Dict[str, List[float]] = {}
        
        logger.info("Aggregator gRPC service initialized")
    
    def GetPublicKey(self, request, context):
        """
        Handle request for the public key.
        
        This method provides clients with the Paillier public key required for encryption.
        
        Args:
            request: GetPublicKeyRequest message
            context: gRPC context
            
        Returns:
            GetPublicKeyResponse with serialized public key
        """
        logger.info("Received public key request")
        
        try:
            # Convert public key components to base64 encoded strings
            n_str = base64.b64encode(str(self.public_key.n).encode()).decode()
            g_str = base64.b64encode(str(self.public_key.g).encode()).decode()
            
            response = pb2.GetPublicKeyResponse(
                n=n_str,
                g=g_str,
                success=True,
                error_message=""
            )
            logger.success("Public key sent successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error providing public key: {e}")
            return pb2.GetPublicKeyResponse(
                n="",
                g="",
                success=False,
                error_message=str(e)
            )
    
    def SubmitUpdate(self, request, context):
        """
        Handle submission of encrypted model updates from clients.
        
        Args:
            request: SubmitUpdateRequest message
            context: gRPC context
            
        Returns:
            SubmitUpdateResponse indicating success or failure
        """
        client_id = request.client_id
        round_id = request.round_id
        
        logger.info(f"Received encrypted update from client {client_id} for round {round_id}")
        
        try:
            # Convert EncryptedValue protos to EncryptedNumber objects
            encrypted_weights = []
            for val in request.encrypted_weights:
                enc_value = base64.b64decode(val.value).decode()
                exponent = int(val.exponent)
                
                # Reconstruct EncryptedNumber (the exact implementation depends on your Paillier library)
                encrypted_val = EncryptedNumber.from_serialized(
                    self.public_key,
                    enc_value,
                    exponent
                )
                encrypted_weights.append(encrypted_val)
            
            # Store the update
            if round_id not in self.client_updates:
                self.client_updates[round_id] = {}
            
            self.client_updates[round_id][client_id] = encrypted_weights
            logger.success(f"Successfully stored update from client {client_id} for round {round_id}")
            
            return pb2.SubmitUpdateResponse(
                success=True,
                message=f"Update received from client {client_id}"
            )
            
        except Exception as e:
            logger.error(f"Error processing update from client {client_id}: {e}")
            return pb2.SubmitUpdateResponse(
                success=False,
                message=f"Error: {str(e)}"
            )
    
    def GetAggregatedModel(self, request, context):
        """
        Provide the aggregated model for a given round.
        
        If the round hasn't been aggregated yet, it will trigger aggregation
        if enough clients have submitted updates.
        
        Args:
            request: GetAggregatedModelRequest message
            context: gRPC context
            
        Returns:
            GetAggregatedModelResponse with aggregated weights
        """
        round_id = request.round_id
        logger.info(f"Received request for aggregated model for round {round_id}")
        
        try:
            # Check if we already have aggregated results for this round
            if round_id in self.aggregation_results:
                logger.info(f"Returning cached aggregation results for round {round_id}")
                aggregated_weights = self.aggregation_results[round_id]
                
                return pb2.GetAggregatedModelResponse(
                    aggregated_weights=aggregated_weights,
                    success=True,
                    message="Aggregated model retrieved successfully",
                    epsilon_spent=self.aggregator_pipeline.dp_mechanism.epsilon
                )
            
            # If not, check if we have enough updates to perform aggregation
            if round_id not in self.client_updates:
                return pb2.GetAggregatedModelResponse(
                    aggregated_weights=[],
                    success=False,
                    message=f"No client updates for round {round_id}",
                    epsilon_spent=0.0
                )
            
            updates = self.client_updates[round_id]
            logger.info(f"Have {len(updates)} client updates for round {round_id}")
            
            # Check minimum number of clients
            min_clients = 2  # This should be configurable
            if len(updates) < min_clients:
                return pb2.GetAggregatedModelResponse(
                    aggregated_weights=[],
                    success=False,
                    message=f"Not enough clients ({len(updates)}/{min_clients}) for round {round_id}",
                    epsilon_spent=0.0
                )
            
            # Perform aggregation
            encrypted_updates = list(updates.values())
            logger.info(f"Performing aggregation for round {round_id} with {len(encrypted_updates)} clients")
            
            aggregated_weights = self.aggregator_pipeline.aggregate_and_finalize(encrypted_updates)
            
            # Store the results
            self.aggregation_results[round_id] = aggregated_weights
            
            # Clean up client updates for this round to save memory
            del self.client_updates[round_id]
            
            logger.success(f"Aggregation completed for round {round_id}")
            
            return pb2.GetAggregatedModelResponse(
                aggregated_weights=aggregated_weights,
                success=True,
                message=f"Aggregated model for round {round_id} with {len(encrypted_updates)} clients",
                epsilon_spent=self.aggregator_pipeline.dp_mechanism.epsilon
            )
            
        except Exception as e:
            logger.error(f"Error generating aggregated model for round {round_id}: {e}")
            return pb2.GetAggregatedModelResponse(
                aggregated_weights=[],
                success=False,
                message=f"Error: {str(e)}",
                epsilon_spent=0.0
            )


def serve(host: str, port: int, public_key: PublicKey, private_key: PrivateKey, 
          aggregator_pipeline, max_workers: int = 10) -> None:
    """
    Start the gRPC server.
    
    Args:
        host: Host address to bind the server.
        port: Port to bind the server.
        public_key: Paillier public key.
        private_key: Paillier private key.
        aggregator_pipeline: Instance of AggregatorPipeline.
        max_workers: Maximum number of threads for the server.
    """
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = AggregatorServicer(public_key, private_key, aggregator_pipeline)
    pb2_grpc.add_AggregatorServiceServicer_to_server(servicer, server)
    
    server_address = f"{host}:{port}"
    server.add_insecure_port(server_address)
    server.start()
    
    logger.success(f"Aggregator gRPC server started on {server_address}")
    
    try:
        # Keep thread alive
        while True:
            time.sleep(86400)  # Sleep for a day
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)
        logger.success("Server stopped gracefully")
