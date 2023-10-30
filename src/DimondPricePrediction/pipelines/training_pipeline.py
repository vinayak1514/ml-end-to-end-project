import os 
import sys 
import pandas as pd 
import numpy as np 
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import CustomException
from src.DimondPricePrediction.components.data_ingestion import DataIngestion
from src.DimondPricePrediction.components.data_transformation import DataTransformation
from src.DimondPricePrediction.components.model_trainer import ModelTrainer

obj=DataIngestion()

train_data_path,test_data_path=obj.initiate_data_ingestion()

data_transformation=DataTransformation()
train_arr , test_arr = data_transformation.initialize_data_transformation(train_data_path,test_data_path)

model = ModelTrainer()
model.initate_model_training(train_arr,test_arr)