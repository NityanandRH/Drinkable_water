import logging
import os
import dill
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import f1_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(xtrain, ytrain, xtest, ytest, models):
    try:
        logging.info("model searching started")
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(xtrain, ytrain)
            y_train_pred = model.predict(xtrain)
            y_test_pred = model.predict(xtest)
            train_model_score = f1_score(ytrain, y_train_pred)
            test_model_score = f1_score(ytest, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e:
        raise CustomException(e, sys)

