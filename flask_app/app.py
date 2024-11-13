from flask import Flask, request, render_template
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Initialize the Flask application
app = Flask(__name__)

# Load your trained model and scaler
model = joblib.load('random_forest_model.pkl')  # replace with your actual model file path if needed
scaler = joblib.load('scaler.pkl')  # replace with your actual scaler file path if needed

# Define expected feature names as per model
expected_feature_names = [
    'GDP per capita', 'Social support', 'Healthy life expectancy',
    'Freedom to make life choices', 'Generosity', 
    'Perceptions of corruption', 'Dystopia + residual'
]

# Define a route for the main page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for predictions with XAI
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = [
        float(request.form['gdp_per_capita']),
        float(request.form['social_support']),
        float(request.form['healthy_life_expectancy']),
        float(request.form['freedom_to_make_life_choices']),  # Updated name here
        float(request.form['generosity']),
        float(request.form['perceptions_of_corruption']),
        float(request.form['dystopia_residual'])  
    ]
    input_df = pd.DataFrame([input_data], columns=expected_feature_names)
    
    # Scale the input data
    scaled_input = scaler.transform(input_df)

    # Make a prediction
    prediction = model.predict(scaled_input)[0]
    
    # Calculate SHAP values using KernelExplainer
    explainer = shap.KernelExplainer(model.predict, shap.sample(scaled_input, 100))
    shap_values = explainer.shap_values(scaled_input)

    # Generate a SHAP plot and convert it to a base64-encoded string
    plt.figure()
    shap.force_plot(explainer.expected_value, shap_values[0], input_df, matplotlib=True, show=False)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    shap_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()

    return render_template('index.html', prediction=prediction, shap_img=shap_img)

if __name__ == '__main__':
    app.run(debug=True)
