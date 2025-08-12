import os
import pickle
import hashlib
import time
from datetime import datetime, timedelta
from loguru import logger

class DataCache:
    """Efficient caching system for stock data and computations"""
    
    def __init__(self, cache_dir='cache', max_age_hours=24):
        self.cache_dir = cache_dir
        self.max_age = timedelta(hours=max_age_hours)
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key):
        """Generate cache file path from key"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.pkl")
    
    def get(self, key):
        """Get cached data if it exists and is not expired"""
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return None
        
        # Check if cache is expired
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - file_time > self.max_age:
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def set(self, key, data):
        """Store data in cache"""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
            return False
        
    def cleanup(self):
        """Remove expired cache files"""
        for filename in os.listdir(self.cache_dir):
            path = os.path.join(self.cache_dir, filename)
            if os.path.isfile(path):
                file_time = datetime.fromtimestamp(os.path.getmtime(path))
                if datetime.now() - file_time > self.max_age:
                    os.remove(path)
                    logger.info(f"Removed expired cache: {filename}")
