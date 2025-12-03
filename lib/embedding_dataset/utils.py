"""
Utility functions for embedding datasets.
"""
import hashlib


def create_deterministic_song_id(clique_str, version_str):
    """Create a truly deterministic ID across all sessions"""
    combined = f"{clique_str}-{version_str}"
    # Use MD5 or SHA256 for true determinism
    hash_bytes = hashlib.md5(combined.encode('utf-8')).digest()
    # Convert first 4 bytes to int (positive 32-bit)
    return int.from_bytes(hash_bytes[:4], byteorder='big') & 0x7fffffff