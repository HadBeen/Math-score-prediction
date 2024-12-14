import numpy as np 
import pandas as pd 
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from data_ingestion import DataIngestion
from sklearn.pipeline import Pipeline
import os
import sys


@dataclass
class DataTransformationConfig :
    preprocesser_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''

        This function does the data transformation.
        
        '''
        try:
            num_features = ['reading_score', 'writing_score']
            cat_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch','test_preparation_course']
            
            num_pipeline = Pipeline(
            steps=[
                ('missing_values_handler',SimpleImputer(strategy='median')),
                ('normalizer',StandardScaler())
                 ]
             )
            
            cat_pipeline = Pipeline(
            steps=[
                ('missing_values_handler',SimpleImputer(strategy='most_frequent')),
                ('categorical_encoder',OneHotEncoder()),
                ('normalizer',StandardScaler())
            ]
             )
            logging.info("num and catcolums stardation completed .")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline",num_pipeline,num_features),
                    ("cat_pipeline",cat_pipeline,cat_features)
                ]
            )
            return preprocessor
        except Exception as e :
            CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Reading train and test data completed.')

            preprocessing_obj = self.get_data_transformer_object()
            target_colum = 'math_score'
            num_features = ['reading_score', 'writing_score']
            input_data_train_df =train_df.drop(columns=[target_colum],axis=1)

        except Exception as e :
            pass




