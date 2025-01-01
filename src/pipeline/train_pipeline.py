import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class TrainPipeline:
    def __init__(self):
        pass
    def train(self,features):
        try :
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path )
            data_scale = preprocessor.transform(features)
            preds = model.predict(data_scale)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        

