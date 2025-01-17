from flask import Flask, request, render_template, send_from_directory
import numpy as np
import os
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Define class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Load the model
model = load_model("my_model(1).keras")  # Ensure the model file path is correct

# Configure upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to predict tumor
def predict_tumor(image_path):
    IMAGE_SIZE = 128  # Expected input size for the model
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img = img_to_array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model input

    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    confidence_score = prediction[0][predicted_class_index]  # Confidence of the predicted class

    if class_labels[predicted_class_index] == "notumor":
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Main route for file upload and prediction
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Check if a file was uploaded
        if 'file' not in request.files or not request.files['file'].filename:
            return render_template('index.html', result="No file uploaded", confidence=None)

        file = request.files['file']
        if file:
            # Save the uploaded file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Perform prediction
            result, confidence = predict_tumor(file_location)

            # Render the result along with the uploaded image
            return render_template(
                'index.html',
                result=result,
                confidence=f"{confidence * 100:.2f}%",
                file_path=f'/uploads/{file.filename}'
            )

    # Render the initial page for GET requests
    return render_template('index.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
