import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
import base64
import numpy as np
from PIL import Image
import io

class ModelHandler:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.class_labels = ['Cendrawasih', 'Kawung', 'Megamendung', 'Sekar', 'Sidomukti']

    def load_model(self, model_path):
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile()
            print(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"Failed to load model from {model_path}. Error: {e}")
            raise

    def decode_base64_image(self, base64_string):
        """
        Decode a base64 string to an image.

        :param base64_string: str, base64 encoded image.
        :return: PIL Image
        """
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image

    def preprocess_image(self, input_image, target_size):
        """
        Preprocess the input image as required by the model.

        :param image: PIL Image, the image to preprocess.
        :param target_size: tuple, target size for the image.
        :return: preprocessed image
        """
        input_image = input_image.resize(target_size)
        img_array = img_to_array(input_image)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image if needed
        return img_array

    def predict(self, base64_string):
        """
        Make a prediction on the input image using the pre-trained model.

        :param base64_string: str, base64 encoded image.
        :return: model prediction
        """
        target_size = (224, 224)  # Change this to the input size your model expects
        image = self.decode_base64_image(base64_string)
        preprocessed_image = self.preprocess_image(image, target_size)
        try:
            prediction = self.model.predict(preprocessed_image)
            predicted_class_index = np.argmax(prediction)  # Get the index of the highest probability
            predicted_class_label = self.class_labels[predicted_class_index]  # Map the index to the class label
            return predicted_class_label
        except Exception as e:
            print(f"Failed to make a prediction. Error: {e}")
            raise
