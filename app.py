from flask import Flask, render_template, request, redirect, make_response
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import os

app = Flask(__name__)

# Load your trained model
model = load_model('traffic_sign_model.h5')

# Labels for the German traffic signs
labels = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
    'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
    'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
    'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
    'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow',
    'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
    'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
    'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            try:
                # Read the image file into memory
                img = Image.open(file.stream).convert("RGB")  # Ensure image is in RGB format
                img = img.resize((50, 50))
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = img / 255.0
                
                # Make prediction
                pred = model.predict(img)
                class_idx = np.argmax(pred)
                class_label = labels[class_idx]
                confidence = pred[0][class_idx] * 100
                
                # Create a response object and set cache control headers
                response = make_response(render_template("result.html", label=class_label, confidence=confidence, image=file.filename))
                response.headers["Cache-Control"] = "no-store"
                return response
            except Exception as e:
                app.logger.error(f"Error processing file: {e}")
                return "An error occurred while processing the image. Please try again."
    
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
