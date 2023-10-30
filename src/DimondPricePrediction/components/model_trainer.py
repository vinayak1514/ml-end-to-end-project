import os 
import sys 
import pandas as pd 
import numpy as np 
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import CustomException
from src.DimondPricePrediction.utils.utils import save_object
# from src.DimondPricePrediction.utils.utils import
from sklearn.linear_model import LinearRegression ,Ridge,Lasso,ElasticNet