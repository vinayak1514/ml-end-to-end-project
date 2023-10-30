import os 
import sys 
import pandas as pd 
import numpy as np 
from dataclasses import dataclass
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import CustomException
from src.DimondPricePrediction.utils.utils import save_object
from src.DimondPricePrediction.utils.utils import evaluate_model
from sklearn.linear_model import LinearRegression ,Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train,y_train,X_test,y_test = [
                    train_array[:,:-1],
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1]
            ]
            models = {
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
            }
            model_report:dict = evaluate_model(X_train,X_test,y_train,y_test,models)
            print(model_report)
            logging.info(f'Model Report : {model_report}')

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            logging.info(f'Best Model Found-Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )



        except Exception as e:
            logging.info('error accured at initate model training ')
            raise CustomException(e,sys)