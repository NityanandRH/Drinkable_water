import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])
            logging.info("Splitting input data into train and test data")
            models = {"Logistic Regression": LogisticRegression(),
                      "Decision Tree": DecisionTreeClassifier(),
                      "XGB Classifier": XGBClassifier(),
                      "SVM": SVC(),
                      "Ada_boost": AdaBoostClassifier(),
                      "Gradient_boost": GradientBoostingClassifier(),
                      "Random_forest": RandomForestClassifier()
                      }
            model_report: dict = evaluate_models(xtrain=X_train, ytrain=y_train, xtest=X_test, ytest=y_test,
                                                 models=models)

            logging.info("Model evaluation Completed")
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            logging.info("Best model found")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            logging.info("Trained Object Saved")

            predicted = best_model.predict(X_test)
            score = f1_score(y_test, predicted)

            return score

        except Exception as e:
            CustomException(e, sys)

