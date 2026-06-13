import os
import pickle

from .....import __CACHE_DIR__

class Tile_Prompter_Cache():

    def cache_path(key, create_dir=False):
        if create_dir:
            os.makedirs(__CACHE_DIR__, exist_ok=True)
        return os.path.join(__CACHE_DIR__, key)

    def isset(key):
        if not os.path.exists(Tile_Prompter_Cache.cache_path(key)):
            return False
        return True

    def set(key, value):
        with open(Tile_Prompter_Cache.cache_path(key, create_dir=True), 'wb') as f:
            pickle.dump(value, f)

    def get(key, default_value=None):
        value = default_value
        if Tile_Prompter_Cache.isset(key):
            try:
                with open(Tile_Prompter_Cache.cache_path(key), 'rb') as f:
                    value = pickle.load(f)
            except FileNotFoundError:
                value = default_value
        return value

    def cache_delete(key):
        cache_path = Tile_Prompter_Cache.cache_path(key)
        if os.path.exists(cache_path):
            os.remove(cache_path)
        return True
