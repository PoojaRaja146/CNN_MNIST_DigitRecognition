from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('model/flower_trained_model_adv.h5')

# Define the index route
@app.route('/')
def index():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    try:
        # Open the image using PIL directly from the file stream
        img = Image.open(file.stream).convert('RGB')
        
        
        # Resize the image to the dimensions expected by the model
        img = img.resize((180,180))
        
        # Convert the image to a NumPy array
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction[0])

        # Map predicted_class index to flower name
        class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
        predicted_label = class_names[predicted_class]

        return render_template('index.html', prediction=predicted_label)
    except Exception as e:
        return str(e)  # Return the error message for debugging

if __name__ == "__main__":
    app.run(debug=True)
