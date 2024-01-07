from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import decode
import numpy as np
from PIL import Image
import io

model = load_model('first_prediction_model.keras')

app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def extract_text():
    image_data = request.files['image'].read()

    image = Image.open(io.BytesIO(image_data))
    image = np.array(image)
    print(image.shape)
    extracted_text = decode.extract_text(image,model)
    print(extracted_text)
    return jsonify({'text': extracted_text}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

