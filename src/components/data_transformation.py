import os
import sys

import numpy as np 
import pandas as pd 
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from sklearn.pipeline import Pipeline
from src.utils import save_object



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
                ('normalizer',StandardScaler(with_mean=False))
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

            input_feature_train_df =train_df.drop(columns=[target_colum],axis=1)
            target_feature_train_df = train_df[target_colum]

            input_feature_test_df =test_df.drop(columns=[target_colum],axis=1)
            target_feature_test_df = test_df[target_colum]

            logging.info("Apply preprocessing to train and test df.")

            input_feature_train_df = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_df = preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_df,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df,np.array(target_feature_test_df)]

            logging.info('Saved preprocessing obj.')

            save_object(
                file_path =  self.data_transformation_config.preprocesser_obj_file_path ,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesser_obj_file_path
            )
        except Exception as e :
            raise CustomException(e,sys)
            




