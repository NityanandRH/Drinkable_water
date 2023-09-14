from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_data', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            ph=request.form.get('ph'),
            hardness=request.form.get('hardness'),
            solids=request.form.get('solids'),
            chloramines=request.form.get('chloramines'),
            sulfate=request.form.get('sulfate'),
            conductivity=request.form.get('conductivity'),
            organic_carbon=request.form.get('organic_carbon'),
            trihalomethanes=request.form.get('trihalomethanes'),
            turbidity=request.form.get('turbidity')
        )
        pred_df = data.get_data_as_df()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
