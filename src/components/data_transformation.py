import os.path
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from src.utils import save_object

from src.exception import CustomException
from src.logger import logging

@dataclass()
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class Datatransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    '''
    This function is responsible for data transformation
    '''
    def get_data_transformer_object(self):
        try:
            columns = ['ph', 'Hardness', 'Solids', 'Chloramines',
                       'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
            num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")),
                                           ("scaler", MinMaxScaler())])
            logging.info("Missing Data handling completed")

            preprocessor = ColumnTransformer([("num_pipeline", num_pipeline, columns)])

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data reading completed")

            preprocessing_obj = self.get_data_transformer_object()
            logging.info("Pre processing object obtained")

            target_colum_name = "Potability"

            input_feature_train_df = train_df.drop(columns=target_colum_name, axis=1)
            target_feature_train_df = train_df[target_colum_name]

            input_feature_test_df = test_df.drop(columns=target_colum_name, axis=1)
            target_feature_test_df = test_df[target_colum_name]

            logging.info("Applying preprocessing on testing and training dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved Preprocessing object")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e,sys)
