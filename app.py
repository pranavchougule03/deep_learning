from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import decode
import numpy as np
from PIL import Image
import io
import base64

model = load_model('first_prediction_model.keras')

app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def extract_text():
    data = request.get_json()  # Receive JSON object

    # Extract Base64-encoded image data
    base64_image_data = data.get('image')

    # Decode Base64 data
    decoded_image_data = base64.b64decode(base64_image_data)

    # Convert decoded data into a PIL image
    image = Image.open(io.BytesIO(decoded_image_data))
    # image = np.array(image)
    if(len(image.shape)==2): #if image is grayscale
        image = np.stack((image,)*3, axis=-1)
        return jsonify({'text': 'grayscale image'}), 200
        # print(image.shape)
    # print(image.shape)
    extracted_text = decode.extract_text(image,model)
    print(extracted_text)
    return jsonify({'text': extracted_text}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

