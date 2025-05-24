"""
Client for interacting with the Aggregator gRPC service.
"""
from typing import Optional, List

import grpc
from loguru import logger

from src.communication.generated import aggregator_service_pb2
from src.communication.generated import aggregator_service_pb2_grpc


class AggregatorClient:
    """
    Client for interacting with the Aggregator gRPC service.
    """
    def __init__(self, host: str = 'localhost', port: int = 50051, timeout: int = 10):
        self.host = host
        self.port = port
        self.channel: Optional[grpc.Channel] = None
        self.stub: Optional[aggregator_service_pb2_grpc.AggregatorServiceStub] = None
        self.timeout = timeout  # Default timeout, can be adjusted

    def connect(self):
        """
        Establishes a connection to the gRPC server.
        """
        try:
            self.channel = grpc.insecure_channel(f'{self.host}:{self.port}')
            # Optional: Wait for the channel to be ready
            # grpc.channel_ready_future(self.channel).result(timeout=self.timeout)
            self.stub = aggregator_service_pb2_grpc.AggregatorServiceStub(self.channel)
            logger.info(
                f"Successfully connected to Aggregator gRPC server at {self.host}:{self.port}"
            )
        except grpc.RpcError as rpc_error:
            logger.error(
                f"Failed to connect to Aggregator gRPC server at {self.host}:{self.port}: {rpc_error}"
            )
            self.channel = None
            self.stub = None

    def get_public_key(self) -> Optional[aggregator_service_pb2.GetPublicKeyResponse]:
        """
        Fetches the public key from the aggregator.
        """
        if not self.stub:
            logger.error("Not connected to aggregator. Call connect() first.")
            return None
        try:
            request = aggregator_service_pb2.GetPublicKeyRequest()
            logger.debug("Sending GetPublicKeyRequest to aggregator...")
            response = self.stub.GetPublicKey(request, timeout=self.timeout)
            logger.success(
                f"Received public key from aggregator (n_size: {len(response.n)} bytes)"
            )
            return response
        except grpc.RpcError as rpc_error:
            logger.error(f"gRPC error getting public key: {rpc_error.code()} - {rpc_error.details()}")
            return None

    def submit_update(
        self, client_id: str, encrypted_weights: List[aggregator_service_pb2.EncryptedValue], round_id: str
    ) -> Optional[aggregator_service_pb2.SubmitUpdateResponse]:
        """
        Submits encrypted model updates to the aggregator.
        Matches the 'SubmitUpdate' RPC in the proto.
        """
        if not self.stub:
            logger.error("Not connected to aggregator. Call connect() first.")
            return None
        try:
            request = aggregator_service_pb2.SubmitUpdateRequest(
                client_id=client_id,
                encrypted_weights=encrypted_weights,
                round_id=round_id
            )
            logger.debug(f"Submitting update for client {client_id}, round {round_id}...")
            response = self.stub.SubmitUpdate(request, timeout=self.timeout)
            logger.success(
                f"Update submitted for client {client_id}. Aggregator: {response.message}"
            )
            return response
        except grpc.RpcError as rpc_error:
            logger.error(f"gRPC error submitting update: {rpc_error.code()} - {rpc_error.details()}")
            return None

    def get_aggregated_model(
        self, round_id: str
    ) -> Optional[aggregator_service_pb2.GetAggregatedModelResponse]:
        """
        Gets the aggregated model from the aggregator.
        Matches the 'GetAggregatedModel' RPC in the proto.
        """
        if not self.stub:
            logger.error("Not connected to aggregator. Call connect() first.")
            return None
        try:
            request = aggregator_service_pb2.GetAggregatedModelRequest(round_id=round_id)
            logger.debug(f"Requesting aggregated model for round {round_id}...")
            response = self.stub.GetAggregatedModel(request, timeout=self.timeout)
            logger.success(
                f"Received aggregated model for round {round_id}. Aggregator: {response.message}"
            )
            return response
        except grpc.RpcError as rpc_error:
            logger.error(f"gRPC error getting aggregated model: {rpc_error.code()} - {rpc_error.details()}")
            return None

    def close(self):
        """
        Closes the gRPC channel.
        """
        if self.channel:
            self.channel.close()
            logger.info("Disconnected from Aggregator gRPC server.")


if __name__ == '__main__':
    # Example Usage (requires aggregator to be running)
    AGGREGATOR_HOST = "localhost"
    AGGREGATOR_PORT = 50051

    client = AggregatorClient(host=AGGREGATOR_HOST, port=AGGREGATOR_PORT)
    client.connect()

    if client.stub:
        # 1. Get Public Key
        pk_response = client.get_public_key()
        if pk_response and pk_response.success:
            logger.success(
                f"Successfully fetched public key. n: {pk_response.n[:30]}..., g: {pk_response.g}"
            )
        else:
            logger.error(f"Failed to fetch public key. Server msg: {pk_response.error_message if pk_response else 'N/A'}")

        # 2. Submit Update (Example)
        # Create dummy EncryptedValue objects. In a real scenario, these would be properly populated.
        dummy_encrypted_weights: List[aggregator_service_pb2.EncryptedValue] = [
            aggregator_service_pb2.EncryptedValue(value="dummy_val1", exponent="dummy_exp1"),
            aggregator_service_pb2.EncryptedValue(value="dummy_val2", exponent="dummy_exp2")
        ]

        if pk_response and pk_response.success:  # Proceed only if public key was obtained
            submit_response = client.submit_update(
                client_id="test_client_001",
                encrypted_weights=dummy_encrypted_weights,
                round_id="round_test_123"
            )
            if submit_response and submit_response.success:
                logger.success(f"Successfully submitted (dummy) update: {submit_response.message}")
            else:
                logger.error(f"Failed to submit update. Server msg: {submit_response.message if submit_response else 'N/A'}")

        # 3. Get Aggregated Model (Example)
        if pk_response and pk_response.success: # Proceed only if public key was obtained
            model_response = client.get_aggregated_model(round_id="round_test_123")
            if model_response and model_response.success:
                logger.success(
                    f"Successfully fetched (dummy) aggregated model: {model_response.message}"
                )
                if model_response.aggregated_weights:
                    logger.info(
                        f"Aggregated weights (first 5): {model_response.aggregated_weights[:5]}"
                    )
            else:
                logger.error(f"Failed to fetch aggregated model. Server msg: {model_response.message if model_response else 'N/A'}")
        
        client.close()
    else:
        logger.error("Could not connect to the aggregator. Example usage aborted.")
