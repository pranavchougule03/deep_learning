import tensorflow as tf
from tensorflow import keras

characters = list(('7', 'y', 'x', '8', 'b', 'n', 'm', '4', '5', '3', 'd', 'p', 'c', 'f', 'g', 'w', 'e', '2', '6'))
char_to_num = keras.layers.StringLookup(vocabulary=list(characters), num_oov_indices=1, mask_token=None)
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
img_height = 50
img_width = 200

def encode_one_for_testing(image):
    print(image.shape)
    img = tf.reduce_mean(image, axis=-1, keepdims=True)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    return tf.expand_dims(img,0)

def decode_batch_predictions(pred):
    input_len = tf.ones(pred.shape[0]) * pred.shape[1]

    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
    results = results[0][0]

    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        result = result.replace("[UNK]", "").strip()
        output_text.append(result)
    return output_text

def extract_text(image,model):
    img = encode_one_for_testing(image)
    pred = model.predict(img)
    return decode_batch_predictions(pred)[0]
