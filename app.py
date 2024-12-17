from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

pipeline_rf = joblib.load('rf_model.pkl')  
label_encoder = joblib.load('label_encoder.pkl')  
  
input_features = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']


@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        inputs = [float(request.form[feature]) for feature in input_features]
        input_array = np.array(inputs).reshape(1, -1)

        
        prediction = pipeline_rf.predict(input_array)
        
       
        predicted_class = label_encoder.inverse_transform(prediction)

        return render_template('result.html', prediction=predicted_class[0])  

if __name__ == '__main__':
    app.run(debug=True)
