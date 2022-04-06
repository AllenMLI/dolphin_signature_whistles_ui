import os
import sys
import cv2
import json
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.utils import Sequence, to_categorical


import dolphin.utils as utils
import dolphin.io_utils as io_utils
from dolphin.models import MODELS
import dolphin.preprocess.feature_extraction as feature_extraction


def generate_features(data_list, savedir, cfg):
    feature_type = cfg["preprocess"]["features"]
    spec_max_length = cfg["preprocess"]["spectrogram_max_length"]

    for data in data_list:
        fp = data.name
        data, sr = librosa.load(data, sr=cfg['preprocess']['sampling_rate'], duration=spec_max_length)

        feature, f, t = feature_extraction.compute_spectrogram(data, sr=sr, cfg=cfg, random_pad=False)
        savename = savedir + fp[:-3] + 'png'
        io_utils.save_fig(feature, f, t, output_dir=savename, cfg=cfg)

    return savedir


class InferenceDataGenerator(Sequence):
    """
    InferenceDataGenerator grabs and loads batches of data.
    """

    def __init__(self, infer_dir):

        self.names = []        
        self.images = []   
        self.visual_purpose = [] 
        for fp in os.listdir(infer_dir):
            img = cv2.imread(infer_dir+fp)
            datapoint = np.expand_dims(img / 255, axis=0)
            self.names.append(fp)
            self.images.append(datapoint)
            self.visual_purpose.append(img)

        self.count = 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        self.count += 1
        return self.images[i], self.names[i]


def run(uploaded_data, model_name, weights, cfg_filename="config.json"):

    with open(cfg_filename, "r") as f:
        cfg = json.load(f)

    # -----------------------------------------------------------------------------------------------------------------
    # Organize the uploaded data
    # -----------------------------------------------------------------------------------------------------------------
    splits = {'inference': []}
    for data in uploaded_data:
        splits['inference'].append(data) 

    # -----------------------------------------------------------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------------------------------------------------------
    inference_dir = 'outputs/ui/classification/spectrograms/'  # for the purpose of running spectrograms through inference
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)

    generate_features(uploaded_data, inference_dir, cfg)
    input_shape = utils.get_input_shape(inference_dir)

    # -----------------------------------------------------------------------------------------------------------------
    # Data Generators
    # -----------------------------------------------------------------------------------------------------------------
    classes = np.sort(['INSERT_CLASS1', 'INSERT_CLASS2', 'INSERT_CLASS3'])
    n_classes = len(classes)
    inference_generator = InferenceDataGenerator(inference_dir)    

    # -----------------------------------------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------------------------------------
    model = MODELS[model_name](include_top=True, weights=weights, input_shape=input_shape, classes=n_classes)
    model.compile(optimizer=optimizers.Adam(learning_rate=cfg["model"]["model_params"]["learning_rate"]),
                  loss='categorical_crossentropy', metrics=['acc'])

    # -----------------------------------------------------------------------------------------------------------------
    # Get them predictions!
    # -----------------------------------------------------------------------------------------------------------------
    predictions = []
    confidences = []
    for img in inference_generator.images:
        output = model.predict(img).squeeze(0)  # get model prediction
        ind = np.argpartition(output, -3)[-3:]  # get indices of top 3 predictions

        confidence = [format(output[ind[2]], '.2%'), format(output[ind[1]], '.2%'), format(output[ind[0]], '.2%')]  # confidence scores of top 3 predictions  
        prediction = [classes[ind[2]], classes[ind[1]], classes[ind[0]]]  # extract the actual classnames of these 3 predictions
        confidence, prediction = (list(t) for t in zip(*sorted(zip(confidence, prediction))))

        predictions.append(prediction[::-1])
        confidences.append(confidence[::-1])   
 
    return predictions, confidences, inference_generator.visual_purpose, inference_generator.names
        


if __name__ == "__main__":
    run()
