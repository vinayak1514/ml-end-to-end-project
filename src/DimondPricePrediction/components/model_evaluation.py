from urllib.parse import urlparse
from src.DimondPricePrediction.utils.utils import load_object

import pickle
import os 
import sys 
import mlflow.sklearn
import numpy as np 
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from src.DimondPricePrediction.exception import CustomException


class ModelEvaluation:
    def __init__(self):
        pass

    def model_evaluation(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual,pred))
        mse = mean_squared_error(actual,pred)
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)
        return rmse,mse,mae,r2

    def initiate_model_evaluation(self,train_array,test_array):
        try:
            X_test,y_test = (test_array[:,:-1],test_array[:,-1])

            model_path = os.path.join('artifacts','model.pkl')
            model  = load_object(model_path)

            mlflow.set_registry_uri('https://dagshub.com/opvinnu02/ml-end-to-end-project.mlflow')
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            print(tracking_url_type_store)
            with mlflow.start_run():
                predicted_qualities = model.predict(X_test)
                (rmse,mse, mae, r2) = self.model_evaluation(y_test,predicted_qualities)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric('mse',mse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

            if tracking_url_type_store!='file':
                mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")

            else:
                    mlflow.sklearn.log_model(model, "model")


        except Exception as e:
            raise CustomException(e,sys)
