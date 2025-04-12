from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Load your classification model
classifier_model =  tf.keras.models.load_model("cnn_pose_classifier.h5")
class_labels = [
    "adho mukh svanasana",
    "ashtanga namaskara",
    "ashwa sanchalanasana",
    "bhujangasana",
    "hasta utthanasana",
    "kumbhakasana",
    "padahastasana",
    "pranamasana"
]
# Load MoveNet from TensorFlow Hub
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet_input_size = 256  # model expects 256x256 input

def detect_pose_movenet(image_bytes):
    # Convert image to numpy array
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((movenet_input_size, movenet_input_size))
    input_image = np.array(image, dtype=np.float32)
    input_image = input_image[np.newaxis, ...]
    input_image = tf.convert_to_tensor(input_image)
    input_image = tf.image.resize_with_pad(input_image, movenet_input_size, movenet_input_size)
    input_image = tf.cast(input_image, dtype=tf.int32)

    # Run MoveNet
    outputs = movenet.signatures['serving_default'](input_image)
    keypoints_with_scores = outputs['output_0'].numpy()
    keypoints = keypoints_with_scores[0, 0, :, :2]  # shape: (17, 2)

    return keypoints.flatten()  # shape: (34,)

@app.route('/predict', methods=['POST'])
def predict():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame uploaded'}), 400

    image_bytes = request.files['frame'].read()

    try:
        coords = detect_pose_movenet(image_bytes)
    except Exception as e:
        return jsonify({'error': f'Pose detection failed: {str(e)}'}), 500

    # Normalize coordinates
    coords = coords / np.linalg.norm(coords)  # L2 normalization

    # Predict pose
    input_tensor = np.expand_dims(coords, axis=0)  # shape (1, 34)
    prediction = classifier_model.predict(input_tensor)
    predicted_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    predicted_class = class_labels[predicted_idx]

    return jsonify({
        'pose': predicted_class,
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
