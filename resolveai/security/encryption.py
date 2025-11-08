"""
Security and Encryption Module

Provides secure encryption, decryption, and key management for ResolveAI.
"""

import os
import base64
import hashlib
import secrets
from typing import Optional, Tuple, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import logging


class EncryptionError(Exception):
    """Custom exception for encryption errors."""
    pass


class EncryptionManager:
    """
    Manages encryption and decryption operations for ResolveAI.
    
    Supports both symmetric encryption (Fernet) and AES encryption
    for different use cases within the application.
    """
    
    def __init__(self, key: Optional[Union[str, bytes]] = None):
        """
        Initialize the encryption manager.
        
        Args:
            key: Optional encryption key. If None, generates a new key.
        """
        self.logger = logging.getLogger(__name__)
        
        if key is None:
            self.key = self._generate_key()
            self.logger.info("Generated new encryption key")
        else:
            self.key = self._prepare_key(key)
        
        # Initialize Fernet for general encryption
        self.fernet = Fernet(self.key)
        
        # For AES encryption (video files)
        self.backend = default_backend()
    
    def _generate_key(self) -> bytes:
        """Generate a new encryption key."""
        return Fernet.generate_key()
    
    def _prepare_key(self, key: Union[str, bytes]) -> bytes:
        """Prepare and validate the encryption key."""
        if isinstance(key, str):
            # Convert string key to bytes
            key_bytes = key.encode('utf-8')
        else:
            key_bytes = key
        
        # Ensure key is valid for Fernet (32 bytes base64-encoded)
        if len(key_bytes) == 44:  # Base64 encoded 32 bytes
            try:
                base64.urlsafe_b64decode(key_bytes)
                return key_bytes
            except Exception:
                pass
        
        # If not a valid Fernet key, derive one
        return self._derive_key_from_password(key_bytes)
    
    def _derive_key_from_password(self, password: bytes, salt: Optional[bytes] = None) -> bytes:
        """
        Derive an encryption key from a password using PBKDF2.
        
        Args:
            password: Password bytes
            salt: Optional salt bytes. If None, generates random salt.
            
        Returns:
            Derived key suitable for Fernet
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=self.backend
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data using Fernet symmetric encryption.
        
        Args:
            data: Data to encrypt (string or bytes)
            
        Returns:
            Base64 encoded encrypted string
        """
        try:
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            encrypted_data = self.fernet.encrypt(data_bytes)
            return encrypted_data.decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt data: {e}")
    
    def decrypt(self, encrypted_data: str) -> bytes:
        """
        Decrypt data using Fernet symmetric encryption.
        
        Args:
            encrypted_data: Base64 encoded encrypted string
            
        Returns:
            Decrypted bytes
        """
        try:
            encrypted_bytes = encrypted_data.encode('utf-8')
            decrypted_data = self.fernet.decrypt(encrypted_bytes)
            return decrypted_data
            
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise EncryptionError(f"Failed to decrypt data: {e}")
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """
        Encrypt a file using AES encryption.
        
        Args:
            file_path: Path to the file to encrypt
            output_path: Optional output path. If None, adds .enc extension
            
        Returns:
            Path to the encrypted file
        """
        try:
            if output_path is None:
                output_path = file_path + ".enc"
            
            # Generate random IV and key for AES
            iv = os.urandom(16)
            aes_key = os.urandom(32)  # 256-bit key
            
            # Create AES cipher
            cipher = Cipher(
                algorithms.AES(aes_key),
                modes.CBC(iv),
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            
            # Read and encrypt file
            with open(file_path, 'rb') as infile:
                file_data = infile.read()
            
            # Pad data to be multiple of 16 bytes
            padded_data = self._pad_data(file_data)
            
            # Encrypt data
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Write encrypted file with IV and key
            with open(output_path, 'wb') as outfile:
                # Store IV and encrypted AES key (encrypted with master key)
                outfile.write(iv)
                encrypted_aes_key = self.fernet.encrypt(aes_key)
                outfile.write(len(encrypted_aes_key).to_bytes(4, 'big'))
                outfile.write(encrypted_aes_key)
                outfile.write(encrypted_data)
            
            self.logger.info(f"File encrypted: {file_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"File encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt file: {e}")
    
    def decrypt_file(self, encrypted_path: str, output_path: Optional[str] = None) -> str:
        """
        Decrypt a file encrypted with AES encryption.
        
        Args:
            encrypted_path: Path to the encrypted file
            output_path: Optional output path. If None, removes .enc extension
            
        Returns:
            Path to the decrypted file
        """
        try:
            if output_path is None:
                if encrypted_path.endswith('.enc'):
                    output_path = encrypted_path[:-4]
                else:
                    output_path = encrypted_path + ".dec"
            
            # Read encrypted file
            with open(encrypted_path, 'rb') as infile:
                # Extract IV
                iv = infile.read(16)
                
                # Extract encrypted AES key
                key_length_bytes = infile.read(4)
                key_length = int.from_bytes(key_length_bytes, 'big')
                encrypted_aes_key = infile.read(key_length)
                
                # Decrypt AES key
                aes_key = self.fernet.decrypt(encrypted_aes_key)
                
                # Extract encrypted data
                encrypted_data = infile.read()
            
            # Create AES cipher
            cipher = Cipher(
                algorithms.AES(aes_key),
                modes.CBC(iv),
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            
            # Decrypt data
            padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
            
            # Remove padding
            file_data = self._unpad_data(padded_data)
            
            # Write decrypted file
            with open(output_path, 'wb') as outfile:
                outfile.write(file_data)
            
            self.logger.info(f"File decrypted: {encrypted_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"File decryption failed: {e}")
            raise EncryptionError(f"Failed to decrypt file: {e}")
    
    def _pad_data(self, data: bytes) -> bytes:
        """Pad data to be multiple of 16 bytes for AES."""
        pad_length = 16 - (len(data) % 16)
        padding = bytes([pad_length] * pad_length)
        return data + padding
    
    def _unpad_data(self, padded_data: bytes) -> bytes:
        """Remove padding from data."""
        pad_length = padded_data[-1]
        return padded_data[:-pad_length]
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure random token.
        
        Args:
            length: Length of the token in bytes
            
        Returns:
            Hex encoded secure token
        """
        return secrets.token_hex(length)
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        Hash a password using PBKDF2.
        
        Args:
            password: Password to hash
            salt: Optional salt. If None, generates random salt.
            
        Returns:
            Tuple of (hashed_password, salt)
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        password_bytes = password.encode('utf-8')
        salt_bytes = bytes.fromhex(salt)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt_bytes,
            iterations=100000,
            backend=self.backend
        )
        
        hashed_password = base64.urlsafe_b64encode(kdf.derive(password_bytes)).decode('utf-8')
        return hashed_password, salt
    
    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """
        Verify a password against a hash.
        
        Args:
            password: Password to verify
            hashed_password: Hashed password to verify against
            salt: Salt used for hashing
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            computed_hash, _ = self.hash_password(password, salt)
            return secrets.compare_digest(computed_hash, hashed_password)
        except Exception:
            return False
    
    def get_key_info(self) -> dict:
        """Get information about the current encryption key."""
        return {
            "key_type": "Fernet (AES-128)",
            "key_length": len(self.key) * 8,  # in bits
            "algorithm": "AES-128-CBC for files, Fernet for data",
            "is_base64": True
        }