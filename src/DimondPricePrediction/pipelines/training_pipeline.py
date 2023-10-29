import os 
import sys 
import pandas as pd 
import numpy as np 
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import CustomException
from src.DimondPricePrediction.components.data_ingestion import DataIngestion
from src.DimondPricePrediction.components.data_transformation import DataTransformation

train = 'D:\study\Data_science\code\machine_learning\end_to_end_project\artifacts\train.csv'
test = 'D:\study\Data_science\code\machine_learning\end_to_end_project\artifacts\test.csv'
obj=DataIngestion()

train_data_path,test_data_path=obj.initiate_data_ingestion()

data_transformation=DataTransformation()
data_transformation.initiate_data_ingestion(train,test)

# train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)