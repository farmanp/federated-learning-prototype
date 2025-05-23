"""
Paillier homomorphic encryption module for secure aggregation.

This module provides functionality for secure aggregation in federated learning using
the Paillier homomorphic encryption scheme, which allows for addition operations on 
encrypted data.
"""

from typing import List, Tuple, Any, Dict, Optional, Union
import numpy as np
from phe import paillier
from loguru import logger

# Type aliases for better readability
PublicKey = paillier.PaillierPublicKey
PrivateKey = paillier.PaillierPrivateKey
EncryptedNumber = paillier.EncryptedNumber
PaillierKeyPair = Tuple[PublicKey, PrivateKey]

class PaillierCrypto:
    """
    A class that provides Paillier homomorphic encryption operations for federated learning.
    
    This class handles key generation, encryption/decryption of weight vectors, and secure
    aggregation of encrypted values.
    """
    
    def __init__(self, key_size: int = 2048):
        """
        Initialize the PaillierCrypto class with a specific key size.
        
        Args:
            key_size: The bit length of the modulus. Larger keys offer more security
                      but are slower to generate and operate with. Default is 2048.
        """
        self.key_size = key_size
        self.public_key = None
        self.private_key = None
    
    def generate_keys(self) -> PaillierKeyPair:
        """
        Generate a new Paillier keypair for encryption and decryption.
        
        Returns:
            A tuple containing (public_key, private_key)
        """
        logger.info(f"Generating Paillier keypair with {self.key_size} bits...")
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=self.key_size)
        logger.success("Keypair generated successfully")
        return self.public_key, self.private_key
    
    def set_keys(self, public_key: Optional[PublicKey] = None, private_key: Optional[PrivateKey] = None) -> None:
        """
        Set existing keys instead of generating new ones.
        
        Args:
            public_key: An existing Paillier public key
            private_key: An existing Paillier private key
        """
        if public_key:
            self.public_key = public_key
        if private_key:
            self.private_key = private_key
        
        # Log what keys were set
        key_status = []
        if self.public_key:
            key_status.append("public")
        if self.private_key:
            key_status.append("private")
        
        if key_status:
            logger.info(f"Set existing Paillier {' and '.join(key_status)} key(s)")
        else:
            logger.warning("No keys were set")
    
    def encrypt_value(self, value: float) -> EncryptedNumber:
        """
        Encrypt a single float value using the public key.
        
        Args:
            value: A float value to encrypt
            
        Returns:
            An encrypted number object
            
        Raises:
            ValueError: If public key is not set
        """
        if not self.public_key:
            logger.error("Cannot encrypt: public key not set")
            raise ValueError("Public key must be set before encryption")
        
        try:
            encrypted = self.public_key.encrypt(value)
            return encrypted
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise
    
    def encrypt_vector(self, vector: List[float]) -> List[EncryptedNumber]:
        """
        Encrypt a vector of float values.
        
        Args:
            vector: A list of float values to encrypt
            
        Returns:
            A list of encrypted number objects
            
        Raises:
            ValueError: If public key is not set
        """
        if not self.public_key:
            logger.error("Cannot encrypt vector: public key not set")
            raise ValueError("Public key must be set before encryption")
        
        try:
            encrypted_vector = [self.public_key.encrypt(value) for value in vector]
            logger.debug(f"Encrypted vector of length {len(vector)}")
            return encrypted_vector
        except Exception as e:
            logger.error(f"Vector encryption error: {e}")
            raise
    
    def decrypt_value(self, encrypted_value: EncryptedNumber) -> float:
        """
        Decrypt a single encrypted value using the private key.
        
        Args:
            encrypted_value: The encrypted number to decrypt
            
        Returns:
            The decrypted float value
            
        Raises:
            ValueError: If private key is not set
            TypeError: If the encrypted value has a different public key
        """
        if not self.private_key:
            logger.error("Cannot decrypt: private key not set")
            raise ValueError("Private key must be set before decryption")
        
        try:
            # Check that the encrypted value uses the same public key
            if encrypted_value.public_key != self.public_key and self.public_key is not None:
                logger.error("Public key mismatch during decryption")
                raise ValueError("Encrypted value was encrypted with a different public key")
            
            decrypted = self.private_key.decrypt(encrypted_value)
            return decrypted
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise
    
    def decrypt_vector(self, encrypted_vector: List[EncryptedNumber]) -> List[float]:
        """
        Decrypt a vector of encrypted values.
        
        Args:
            encrypted_vector: A list of encrypted numbers to decrypt
            
        Returns:
            A list of decrypted float values
            
        Raises:
            ValueError: If private key is not set
        """
        if not self.private_key:
            logger.error("Cannot decrypt vector: private key not set")
            raise ValueError("Private key must be set before decryption")
        
        try:
            decrypted_vector = [self.private_key.decrypt(value) for value in encrypted_vector]
            logger.debug(f"Decrypted vector of length {len(encrypted_vector)}")
            return decrypted_vector
        except Exception as e:
            logger.error(f"Vector decryption error: {e}")
            raise

def aggregate_encrypted_vectors(encrypted_vectors: List[List[EncryptedNumber]]) -> List[EncryptedNumber]:
    """
    Aggregate multiple encrypted vectors element-wise.
    
    Args:
        encrypted_vectors: A list of lists, where each inner list is a vector of encrypted numbers
        
    Returns:
        A list of encrypted numbers representing the element-wise sum
        
    Raises:
        ValueError: If the vectors have different lengths or are empty
    """
    if not encrypted_vectors:
        logger.error("No vectors provided for aggregation")
        raise ValueError("Cannot aggregate empty list of vectors")
    
    # Check that all vectors have the same length
    vector_length = len(encrypted_vectors[0])
    if not all(len(vec) == vector_length for vec in encrypted_vectors):
        logger.error("Vectors have different lengths, cannot aggregate")
        raise ValueError("All vectors must have the same length for aggregation")
    
    try:
        # Initialize the result with the first vector
        result = encrypted_vectors[0].copy()
        
        # Add the remaining vectors
        for i in range(1, len(encrypted_vectors)):
            for j in range(vector_length):
                result[j] += encrypted_vectors[i][j]
        
        logger.success(f"Successfully aggregated {len(encrypted_vectors)} encrypted vectors of length {vector_length}")
        return result
    except Exception as e:
        logger.error(f"Aggregation error: {e}")
        raise


# Convenience functions that don't require class instantiation

def generate_keys(key_size: int = 2048) -> PaillierKeyPair:
    """
    Generate a new Paillier keypair with the specified key size.
    
    Args:
        key_size: The bit length of the modulus
        
    Returns:
        A tuple containing (public_key, private_key)
    """
    crypto = PaillierCrypto(key_size=key_size)
    return crypto.generate_keys()

def encrypt_vector(vector: List[float], public_key: PublicKey) -> List[EncryptedNumber]:
    """
    Encrypt a vector using the provided public key.
    
    Args:
        vector: A list of float values to encrypt
        public_key: The Paillier public key to use for encryption
        
    Returns:
        A list of encrypted numbers
    """
    crypto = PaillierCrypto()
    crypto.set_keys(public_key=public_key)
    return crypto.encrypt_vector(vector)

def decrypt_vector(encrypted_vector: List[EncryptedNumber], private_key: PrivateKey) -> List[float]:
    """
    Decrypt a vector using the provided private key.
    
    Args:
        encrypted_vector: A list of encrypted numbers
        private_key: The Paillier private key to use for decryption
        
    Returns:
        A list of decrypted float values
    """
    crypto = PaillierCrypto()
    crypto.set_keys(private_key=private_key)
    return crypto.decrypt_vector(encrypted_vector)
