from flask import Flask, render_template, Response, send_from_directory
import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

# Load the trained model
model_path = 'densenet201_trashbox_model.h5'
model = load_model(model_path)

# Create a list of class labels
train_path = r"C:\Users\DELL\TrashBox\train"
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
class_labels = list(train_generator.class_indices.keys())

# Function to classify the object
def classify_object():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        # Check if the frame is successfully retrieved
        if not ret:
            print("Failed to retrieve frame.")
            continue

        img = cv2.resize(frame, (224, 224))
        img_for_prediction = image.img_to_array(img)
        img_for_prediction = np.expand_dims(img_for_prediction, axis=0)
        img_for_prediction = preprocess_input(img_for_prediction)
        predictions = model.predict(img_for_prediction)
        predicted_class = class_labels[np.argmax(predictions)]

        # Get coordinates for the bounding box
        x, y, w, h = 50, 50, 150, 150  # Adjust these values based on your detection

        # Draw bounding box on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the predicted class
        cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('temp.html')

@app.route('/video_feed')
def video_feed():
    return Response(classify_object(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)


# Before you run the Python code snippet below, run the following command:
# pip install roboflow autodistill autodistill_grounded_sam scikit-learn


