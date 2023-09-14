import sys
import pandas as pd
from src.exception import CustomException

from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaling = preprocessor.transform(features)
            preds = model.predict(data_scaling)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity):
        self.ph = ph
        self.hardness = hardness
        self.solids = solids
        self.chloramines = chloramines
        self.sulfate = sulfate
        self.conductivity = conductivity
        self.organic_carbon = organic_carbon
        self.trihalomethanes = trihalomethanes
        self.turbidity = turbidity

    def get_data_as_df(self):
        try:
            custom_data_input_dict = {
                'ph': [self.ph],
                'hardness': [self.hardness],
                'solids': [self.solids],
                'chloramines': [self.chloramines],
                'sulfate': [self.sulfate],
                'conductivity': [self.conductivity],
                'organic_carbon': [self.organic_carbon],
                'trihalomethanes': [self.trihalomethanes],
                'turbidity': [self.turbidity]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)



