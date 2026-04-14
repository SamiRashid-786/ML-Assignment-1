import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Load your trained brain
print("Waking up the AI...")
model = load_model('my_gender_model.h5')
print("Model loaded successfully!")

# 2. The brain's processing function
def predict_gender(image):
    # If no image is uploaded, don't crash.
    if image is None:
        return "Please upload an image."
        
    # Squeeze and process the image exactly like we did in training
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    img = cv2.resize(img, (64, 64)) 
    img = img / 255.0 
    img = img.reshape(1, 64, 64, 1) 
    
    # Make the guess
    prediction = model.predict(img)[0][0]
    
    # Read the probability
    if prediction > 0.5:
        return f"Female (Confidence: {prediction * 100:.2f}%)"
    else:
        return f"Male (Confidence: {(1 - prediction) * 100:.2f}%)"

# 3. The Bitchass UI
ui = gr.Interface(
    fn=predict_gender, 
    inputs=gr.Image(type="numpy", label="Upload a Face"), 
    outputs=gr.Text(label="AI Prediction"), 
    title="Sami's Gender Classification AI",
    description="Upload a picture of a face, and the Convolutional Neural Network will predict the gender."
)

# Launch the local server
ui.launch()