import os
import sys
import pickle
from src.exception import Custom_Exp
from src.logger import logging

def save_object(file_path, obj):
    try:
        path = os.path.dirname(file_path)
        os.makedirs(path, exist_ok = True)

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

    except Exception as e:
        logging.error(f"Fatal error has occured while saving object to {file_path}")
        raise Custom_Exp(e, sys)