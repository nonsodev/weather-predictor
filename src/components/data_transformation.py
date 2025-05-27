import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.app_logging.logger import logging
from src.utils.utils import save_pkl_file, load_pkl_file



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts", "preprocessor.pkl")
    label_encoder_obj_file_path = os.path.join("artifacts", "le.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.preprocessor = self.get_data_transformer_object()
    
    def get_data_transformer_object(self):
        logging.info("getting preprocessor")
        try:
            numerical_cols = ["Temperature","Humidity","Wind Speed","Precipitation (%)","Atmospheric Pressure","UV Index","Visibility (km)"]
            categorical_cols = ["Cloud Cover","Season","Location"]

            numerical_pipeline = Pipeline([("scaler",StandardScaler(with_mean=False))])
            categorical_pipeline = Pipeline([("encoder",OneHotEncoder())])

            preprocessor = ColumnTransformer([("numerical pipeline", numerical_pipeline, numerical_cols), ("categorical",categorical_pipeline, categorical_cols)])
            logging.info("saved and returning preprocessor")
            return preprocessor
        
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def preprocess_data(self, train_path, test_path):
        try:
            logging.info("began preprocessing data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            train_input_df = train_df.drop("Weather Type", axis=1)
            test_input_df = test_df.drop("Weather Type", axis=1)
            
            train_output_series = train_df["Weather Type"]
            test_output_series = test_df["Weather Type"]
            
            
            le = LabelEncoder()
            train_output_arr = le.fit_transform(train_output_series)
            test_output_arr = le.transform(test_output_series)
            
            # preprocessor = load_pkl_file(self.data_transformation_config.preprocessor_obj_file_path)
            preprocessor = self.preprocessor
            train_input_arr = preprocessor.fit_transform(train_input_df)
            test_input_arr = preprocessor.transform(test_input_df)
            logging.info("splitted and preprocessed")
            
            train_arr = np.c_[train_input_arr, train_output_arr]
            test_arr  =np.c_[test_input_arr, test_output_arr]
            logging.info("joined the input and output together")
            
            save_pkl_file(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)
            save_pkl_file(self.data_transformation_config.label_encoder_obj_file_path, le)
            return (
                train_arr,
                test_arr
            )
        except Exception as e:
            raise CustomException(e, sys)
            
            