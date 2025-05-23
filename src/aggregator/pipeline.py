"""
Aggregator Pipeline - Server-side secure aggregation with Differential Privacy.

This module implements the core aggregation logic that:
1. Accepts encrypted updates from multiple clients
2. Aggregates them securely using Paillier homomorphic encryption
3. Adds differential privacy noise post-aggregation
4. Decrypts and returns the final weights
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from loguru import logger

from src.smc.paillier import (
    EncryptedNumber, 
    PublicKey, 
    PrivateKey,
    aggregate_encrypted_vectors,
    decrypt_vector
)
from src.dp.gaussian_mechanism import GaussianMechanism


class AggregatorPipeline:
    """
    Server-side aggregator that combines secure multi-party computation with differential privacy.
    
    This class orchestrates the complete aggregation process:
    - Receiving encrypted model updates from clients
    - Securely aggregating them using Paillier homomorphic encryption
    - Adding calibrated noise for differential privacy
    - Returning the final decrypted, privacy-preserving weights
    """
    
    def __init__(
        self, 
        public_key: PublicKey, 
        private_key: PrivateKey,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        dp_sensitivity: float = 1.0
    ):
        """
        Initialize the aggregator pipeline.
        
        Args:
            public_key: Paillier public key for encryption operations.
            private_key: Paillier private key for decryption operations.
            dp_epsilon: Privacy parameter (smaller = more privacy).
            dp_delta: Probability of privacy breach.
            dp_sensitivity: Maximum influence one client can have on the output.
        """
        self.public_key = public_key
        self.private_key = private_key
        
        # Initialize the differential privacy mechanism
        self.dp_mechanism = GaussianMechanism(
            epsilon=dp_epsilon,
            delta=dp_delta,
            sensitivity=dp_sensitivity
        )
        
        self.aggregation_count = 0
        logger.info(
            f"Aggregator pipeline initialized with DP parameters: "
            f"ε={dp_epsilon}, δ={dp_delta}, sensitivity={dp_sensitivity}"
        )
    
    def aggregate_and_finalize(
        self, 
        encrypted_updates: List[List[EncryptedNumber]],
        dp_params: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Main aggregation function that performs the complete pipeline.
        
        This function:
        1. Validates the encrypted updates
        2. Aggregates them using Paillier homomorphic addition
        3. Decrypts the aggregated result
        4. Adds differential privacy noise
        5. Returns the final privacy-preserving weights
        
        Args:
            encrypted_updates: List of encrypted weight vectors from clients.
                              Format: List[List[EncryptedNumber]]
            dp_params: Optional dictionary to override DP parameters for this round.
                      Keys: 'epsilon', 'delta', 'sensitivity'
        
        Returns:
            List[float]: Final decrypted weights with differential privacy protection.
            
        Raises:
            ValueError: If input validation fails.
            RuntimeError: If aggregation or decryption fails.
        """
        self.aggregation_count += 1
        num_clients = len(encrypted_updates)
        
        logger.info(f"Starting aggregation round {self.aggregation_count} with {num_clients} clients")
        
        # Step 1: Validate inputs
        self._validate_encrypted_updates(encrypted_updates)
        
        # Step 2: Update DP parameters if provided
        if dp_params:
            self._update_dp_parameters(dp_params, num_clients)
        
        try:
            # Step 3: Secure aggregation using Paillier homomorphic encryption
            logger.info("Performing secure aggregation of encrypted updates...")
            aggregated_encrypted = aggregate_encrypted_vectors(encrypted_updates)
            logger.success(f"Successfully aggregated {num_clients} encrypted vectors")
            
            # Step 4: Decrypt the aggregated result
            logger.info("Decrypting aggregated weights...")
            aggregated_weights = decrypt_vector(aggregated_encrypted, self.private_key)
            logger.success(f"Decrypted aggregated vector of length {len(aggregated_weights)}")
            
            # Step 5: Add differential privacy noise
            logger.info("Adding differential privacy noise...")
            private_weights = self.dp_mechanism.add_noise(aggregated_weights)
            logger.success(f"Added DP noise with σ={self.dp_mechanism.sigma:.6f}")
            
            # Step 6: Log final results for monitoring
            self._log_aggregation_results(aggregated_weights, private_weights, num_clients)
            
            return private_weights
            
        except Exception as e:
            logger.error(f"Aggregation pipeline failed: {e}")
            raise RuntimeError(f"Failed to complete aggregation: {e}")
    
    def _validate_encrypted_updates(self, encrypted_updates: List[List[EncryptedNumber]]) -> None:
        """
        Validate the structure and consistency of encrypted updates.
        
        Args:
            encrypted_updates: List of encrypted weight vectors from clients.
            
        Raises:
            ValueError: If validation fails.
        """
        if not encrypted_updates:
            raise ValueError("No encrypted updates provided")
        
        # For normal operation we require at least 2 clients, but allow single client for testing
        if len(encrypted_updates) < 1:
            raise ValueError("At least 1 client update required")
        
        # Warn if only a single client is used (not ideal for privacy)
        if len(encrypted_updates) == 1:
            logger.warning("Only one client update provided - minimal privacy protection")
        
        # Check that all vectors have the same length
        vector_length = len(encrypted_updates[0])
        if not all(len(update) == vector_length for update in encrypted_updates):
            raise ValueError("All client updates must have the same vector length")
        
        # Check that we have valid encrypted numbers
        for i, update in enumerate(encrypted_updates):
            if not update:  # Empty vector
                raise ValueError(f"Client {i} provided empty update vector")
            
            for j, encrypted_val in enumerate(update):
                if not isinstance(encrypted_val, EncryptedNumber):
                    raise ValueError(f"Client {i}, position {j}: not a valid EncryptedNumber")
                
                # Verify the encrypted value uses our public key
                if encrypted_val.public_key != self.public_key:
                    raise ValueError(f"Client {i}, position {j}: encrypted with different public key")
        
        logger.debug(f"Validated {len(encrypted_updates)} client updates, vector length: {vector_length}")
    
    def _update_dp_parameters(self, dp_params: Dict[str, Any], num_clients: int) -> None:
        """
        Update differential privacy parameters for this aggregation round.
        
        Args:
            dp_params: Dictionary containing DP parameter overrides.
            num_clients: Number of clients participating in this round.
        """
        epsilon = dp_params.get('epsilon')
        delta = dp_params.get('delta')
        sensitivity = dp_params.get('sensitivity')
        
        # Adjust sensitivity based on number of clients if not explicitly provided
        if sensitivity is None and num_clients > 0:
            # For averaging aggregation, sensitivity typically scales inversely with number of clients
            sensitivity = self.dp_mechanism.sensitivity / num_clients
            logger.info(f"Auto-adjusted sensitivity to {sensitivity:.6f} for {num_clients} clients")
        
        # Update the DP mechanism
        self.dp_mechanism.update_parameters(
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity
        )
        
        logger.info(f"Updated DP parameters: ε={self.dp_mechanism.epsilon}, "
                   f"δ={self.dp_mechanism.delta}, sensitivity={self.dp_mechanism.sensitivity}")
    
    def _log_aggregation_results(
        self, 
        aggregated_weights: List[float], 
        private_weights: List[float], 
        num_clients: int
    ) -> None:
        """
        Log aggregation results for monitoring and debugging.
        
        Args:
            aggregated_weights: Weights before adding DP noise.
            private_weights: Final weights after adding DP noise.
            num_clients: Number of participating clients.
        """
        # Calculate statistics
        aggregated_mean = np.mean(aggregated_weights)
        aggregated_std = np.std(aggregated_weights)
        private_mean = np.mean(private_weights)
        private_std = np.std(private_weights)
        
        # Calculate noise impact
        noise_impact = np.mean(np.abs(np.array(private_weights) - np.array(aggregated_weights)))
        
        logger.info("=== Aggregation Results ===")
        logger.info(f"Clients: {num_clients}")
        logger.info(f"Vector length: {len(aggregated_weights)}")
        logger.info(f"Aggregated weights - Mean: {aggregated_mean:.6f}, Std: {aggregated_std:.6f}")
        logger.info(f"Private weights - Mean: {private_mean:.6f}, Std: {private_std:.6f}")
        logger.info(f"Noise impact (MAE): {noise_impact:.6f}")
        logger.info(f"DP parameters - ε: {self.dp_mechanism.epsilon}, δ: {self.dp_mechanism.delta}")
        logger.info(f"Noise scale (σ): {self.dp_mechanism.sigma:.6f}")
        logger.info("========================")
    
    def get_privacy_budget_spent(self) -> Dict[str, Any]:
        """
        Get information about privacy budget spent across aggregation rounds.
        
        Returns:
            Dictionary containing privacy budget information.
        """
        # Simple composition analysis (conservative upper bound)
        effective_epsilon = (
            self.dp_mechanism.epsilon * 
            np.sqrt(2 * self.aggregation_count * np.log(1/self.dp_mechanism.delta))
        )
        effective_delta = self.dp_mechanism.delta * self.aggregation_count
        
        return {
            "rounds_completed": self.aggregation_count,
            "per_round_epsilon": self.dp_mechanism.epsilon,
            "per_round_delta": self.dp_mechanism.delta,
            "effective_epsilon": effective_epsilon,
            "effective_delta": effective_delta,
            "noise_scale": self.dp_mechanism.sigma
        }


def aggregate_and_finalize(
    encrypted_updates: List[List[EncryptedNumber]],
    public_key: PublicKey,
    private_key: PrivateKey,
    dp_params: Dict[str, Any]
) -> List[float]:
    """
    Convenience function for one-shot aggregation.
    
    This function provides a simple interface for performing secure aggregation
    with differential privacy without needing to manage an AggregatorPipeline instance.
    
    Args:
        encrypted_updates: List of encrypted weight vectors from clients.
        public_key: Paillier public key.
        private_key: Paillier private key.
        dp_params: Differential privacy parameters.
                  Keys: 'epsilon', 'delta', 'sensitivity'
    
    Returns:
        List[float]: Final decrypted weights with differential privacy protection.
    """
    # Extract DP parameters
    epsilon = dp_params.get('epsilon', 1.0)
    delta = dp_params.get('delta', 1e-5)
    sensitivity = dp_params.get('sensitivity', 1.0)
    
    # Create a temporary pipeline
    pipeline = AggregatorPipeline(
        public_key=public_key,
        private_key=private_key,
        dp_epsilon=epsilon,
        dp_delta=delta,
        dp_sensitivity=sensitivity
    )
    
    # Perform aggregation
    return pipeline.aggregate_and_finalize(encrypted_updates)
