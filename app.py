from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('model/saved_model.h5')

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
        img = Image.open(file.stream)
        
        # Convert image to grayscale if it's not already
        img = img.convert('L')
        
        # Resize the image to the dimensions expected by the model
        img = img.resize((28, 28))
        
        # Convert the image to a NumPy array
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        img_array = img_array / 255.0  # Normalize

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction[0])

        return render_template('index.html', prediction=predicted_class)
    except Exception as e:
        return str(e)  # Return the error message for debugging

if __name__ == "__main__":
    app.run(debug=True)
