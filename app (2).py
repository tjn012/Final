import subprocess
import sys

# Force install scikit-learn if not found
try:
    import sklearn
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    import sklearn  # Import again after installation

import gradio as gr
import pandas as pd
import pickle

# Load the pre-trained model
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

def predict_HTL_yield(Biomass type, C (wt%), H (wt%), N (wt%), Temprature (°C), Residence time (min), Pressure (MPa)):
    # Creating input DataFrame for the model
    input_data = pd.DataFrame({
        'Token_0': [Biomass type],
        'Token_1': [C (wt%)],
        'Token_2': [H (wt%)],
        'Token_3': [N (wt%)],
        'Token_4': [Temprature (°C)],
        'Token_5': [Residence time (min)],
        'Token_6': [Pressure (MPa)],
    })

    # One-hot encode the input data (ensure it matches the training data)
    input_encoded = pd.get_dummies(input_data)

    # Align columns with the training data (required columns)
    required_columns = model.feature_names_in_  # Get the feature columns from the model
    for col in required_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[required_columns]

    # Make the prediction
    prediction = model.predict(input_encoded)[0]

    # Reverse the label encoding (map the prediction back to the coffee type)
    HTL_yield = label_encoder.inverse_transform([prediction])[0]

    return HTL_yield

# Gradio Interface using components
interface = gr.Interface(
    fn=predict_HTL_yield,
    inputs=[
        gr.Dropdown(['Spirulina', 'Pinus'], label="Biomass type"),
        gr.Dropdown(['low', 'high'=], label="C (wt%)"),
        gr.Dropdown(['low', 'high'], label="H (wt%)"),
        gr.Dropdown(['low', 'high'], label="N (wt%)"),
        gr.Dropdown(['hot', 'cold'], label="Temprature (°C)"),
        gr.Dropdown(['high', 'low'], label="Residence time (min)"),
        gr.Dropdown(['low', 'medium', 'high'], label="Pressure (MPa)"),
    ],
    outputs=gr.Textbox(label="HTL Yield Inputs"),
    title="HTL Yield Prediction"
)

if __name__ == "__main__":
    interface.launch()