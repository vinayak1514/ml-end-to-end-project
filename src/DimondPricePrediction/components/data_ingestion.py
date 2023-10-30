import sys 
import os 
import pandas as pd 
import numpy as np 
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from pathlib import Path
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import CustomException



class DataIngestionConfig:
    raw_data_path:str = os.path.join('artifacts','raw.csv')
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion start')
        try:
            data = pd.read_csv(Path(os.path.join('notebooks\data','gemstone.csv')))
            logging.info('Read dataset as DataFrame')

            os.makedirs(os.path.dirname(os.path.join(self.data_ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            logging.info('Saved the raw data in artifacts folder')

            train_data,test_data = train_test_split(data,test_size=0.25)
            logging.info("train test split completed")

            train_data.to_csv(self.data_ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.data_ingestion_config.test_data_path,index=False)
            logging.info('Data ingestion completed')
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info('error in data ingestion part')
            raise CustomException(e,sys)