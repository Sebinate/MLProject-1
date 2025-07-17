import os
import sys
import pandas as pd
from src.exception import Custom_Exp
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    raw_data_path: str = os.path.join('artifact', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Initializing data ingestion method")

        try:
            df = pd.read_csv(r'notebook\data\stud.csv')
            logging.info('Data has been read as pandas dataframe')

            logging.info('Creating Directories')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            logging.info("Saving raw data file")
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info("Initiating Train-Test split")
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)
            
            logging.info("Saving train set and test set")
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )

        except Exception as e:
            raise Custom_Exp(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()