import sys
sys.path.insert(0, "/home/citao/github/tensorflow-open_nsfw/")

import numpy as np
import tensorflow as tf
from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader

IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"


class nsfw_detector():
    def __init__(self,
        model_weights = '/home/citao/github/tensorflow-open_nsfw/data/open_nsfw-weights.npy',
        image_loader = 'IMAGE_LOADER_YAHOO',
        input_type = InputType.TENSOR.name.lower()
    ):
        self._sess = tf.Session()
        self._model = OpenNsfwModel()
        input_type = InputType[input_type.upper()]
        self._model.build(weights_path = model_weights,
                          input_type = input_type)
        
        self.fn_load_image = None
        if input_type == InputType.TENSOR:
            if image_loader == IMAGE_LOADER_TENSORFLOW:
                self.fn_load_image = create_tensorflow_image_loader(tf.Session(graph=tf.Graph()))
            else:
                self.fn_load_image = create_yahoo_image_loader()
        elif input_type == InputType.BASE64_JPEG:
            import base64
            self.fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])

        self._sess.run(tf.global_variables_initializer())

    def predict(self, input_file):
        image = self.fn_load_image(input_file)
        predictions = self._sess.run(
            self._model.predictions,
            feed_dict = {self._model.input: image}
        )
        probs = predictions[0]
        result = {
            'SFW': probs[0],
            'NSFW': probs[1]
        }
        return result
