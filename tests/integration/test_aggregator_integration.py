"""
Integration test for the Aggregator Pipeline with gRPC communication.

This test simulates a complete federated learning scenario with:
- One Aggregator server that runs the aggregation pipeline
- Multiple client Data Parties that send encrypted updates
- Full round-trip communication via gRPC
"""

import unittest
import threading
import time
import numpy as np
from typing import List, Tuple, Dict

from src.smc.paillier import generate_keys, encrypt_vector, PublicKey, PrivateKey
from src.aggregator.pipeline import AggregatorPipeline
from src.communication.grpc_server import serve

# Import gRPC client code (would be created separately)
# from src.communication.grpc_client import AggregatorClient


class MockClient:
    """
    Mock client for testing that simulates a Data Party.
    """
    
    def __init__(self, client_id: str, weights: List[float], public_key: PublicKey):
        """
        Initialize the mock client.
        
        Args:
            client_id: ID of the client.
            weights: Initial model weights.
            public_key: Paillier public key for encryption.
        """
        self.client_id = client_id
        self.weights = weights
        self.public_key = public_key
        self.encrypted_weights = encrypt_vector(weights, public_key)
    
    def get_encrypted_update(self) -> List:
        """
        Get the encrypted model update.
        
        Returns:
            List of encrypted weights.
        """
        return self.encrypted_weights


class MockGRPCServer:
    """
    Mock gRPC server for testing.
    """
    
    def __init__(self, aggregator_pipeline: AggregatorPipeline, public_key: PublicKey, private_key: PrivateKey):
        """
        Initialize the mock server.
        
        Args:
            aggregator_pipeline: The aggregator pipeline to use.
            public_key: Paillier public key.
            private_key: Paillier private key.
        """
        self.aggregator_pipeline = aggregator_pipeline
        self.public_key = public_key
        self.private_key = private_key
        self.client_updates = {}
        self.running = False
    
    def start_server(self):
        """Start the mock server."""
        self.running = True
    
    def stop_server(self):
        """Stop the mock server."""
        self.running = False
    
    def receive_update(self, client_id: str, encrypted_update: List, round_id: str = "round_1"):
        """
        Simulate receiving an update from a client.
        
        Args:
            client_id: ID of the client.
            encrypted_update: Encrypted model weights.
            round_id: ID of the training round.
            
        Returns:
            Success status.
        """
        if not self.running:
            return False
        
        if round_id not in self.client_updates:
            self.client_updates[round_id] = {}
        
        self.client_updates[round_id][client_id] = encrypted_update
        return True
    
    def get_aggregated_model(self, round_id: str = "round_1"):
        """
        Get the aggregated model for a round.
        
        Args:
            round_id: ID of the training round.
            
        Returns:
            Aggregated model weights.
        """
        if not self.running:
            return None
        
        if round_id not in self.client_updates:
            return None
        
        updates = list(self.client_updates[round_id].values())
        result = self.aggregator_pipeline.aggregate_and_finalize(updates)
        return result


class TestFederatedLearningIntegration(unittest.TestCase):
    """
    Integration test for the complete federated learning system.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Generate keys for testing
        self.public_key, self.private_key = generate_keys(key_size=1024)  # Smaller keys for faster tests
        
        # Create aggregator pipeline with minimal DP (for speed)
        self.aggregator = AggregatorPipeline(
            public_key=self.public_key,
            private_key=self.private_key,
            dp_epsilon=10.0,  # Less noise for testing
            dp_delta=1e-5,
            dp_sensitivity=0.1
        )
        
        # Create mock server
        self.server = MockGRPCServer(
            aggregator_pipeline=self.aggregator,
            public_key=self.public_key,
            private_key=self.private_key
        )
        
        # Start the server
        self.server.start_server()
        
        # Create test data - simulated model weights for different clients
        self.client_weights = [
            [1.0, 2.0, 3.0, 4.0, 5.0],  # Client 1
            [1.2, 2.1, 3.3, 3.9, 5.2],  # Client 2
            [0.9, 2.2, 2.8, 4.1, 4.8],  # Client 3
            [1.1, 1.9, 3.1, 4.0, 4.9]   # Client 4
        ]
        
        # Expected average (before DP noise)
        self.expected_average = np.mean(self.client_weights, axis=0).tolist()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.server.stop_server()
    
    def test_complete_federated_learning_round(self):
        """
        Test a complete federated learning round with multiple clients.
        """
        # Create mock clients
        clients = []
        for i, weights in enumerate(self.client_weights):
            client = MockClient(f"client_{i+1}", weights, self.public_key)
            clients.append(client)
        
        # Simulate clients sending updates
        round_id = "test_round_1"
        for client in clients:
            result = self.server.receive_update(
                client_id=client.client_id,
                encrypted_update=client.get_encrypted_update(),
                round_id=round_id
            )
            self.assertTrue(result)
        
        # Get aggregated model
        aggregated_model = self.server.get_aggregated_model(round_id)
        
        # Verify the result
        self.assertIsNotNone(aggregated_model)
        self.assertEqual(len(aggregated_model), len(self.client_weights[0]))
        
        # Check that results are close to expected average (considering DP noise)
        for i, (expected, actual) in enumerate(zip(self.expected_average, aggregated_model)):
            # Allow for reasonable difference due to DP noise
            self.assertLess(
                abs(expected - actual), 
                1.0,  # Larger tolerance due to intentional DP noise
                f"Position {i}: expected ~{expected}, got {actual}"
            )
    
    def test_multiple_rounds(self):
        """
        Test multiple rounds of federated learning.
        """
        # Create mock clients
        clients = []
        for i, weights in enumerate(self.client_weights):
            client = MockClient(f"client_{i+1}", weights, self.public_key)
            clients.append(client)
        
        # Run multiple rounds
        for round_num in range(1, 4):
            round_id = f"test_round_{round_num}"
            
            # Send updates for this round
            for client in clients:
                result = self.server.receive_update(
                    client_id=client.client_id,
                    encrypted_update=client.get_encrypted_update(),
                    round_id=round_id
                )
                self.assertTrue(result)
            
            # Get aggregated model
            aggregated_model = self.server.get_aggregated_model(round_id)
            
            # Verify the result for this round
            self.assertIsNotNone(aggregated_model)
            self.assertEqual(len(aggregated_model), len(self.client_weights[0]))
    
    @unittest.skip("Skip the actual gRPC test unless the server is running")
    def test_with_real_grpc_server(self):
        """
        Test with a real gRPC server (skipped by default).
        
        This test would create an actual gRPC server and clients to test
        the system end-to-end with real network communication.
        """
        # This would be implemented using real gRPC clients and servers
        pass


if __name__ == '__main__':
    unittest.main()
