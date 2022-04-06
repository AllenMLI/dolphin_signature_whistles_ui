import os
import sys
import cv2
import json
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.utils import Sequence, to_categorical

sys.path.insert(1, os.path.join(sys.path[0], 'src'))
import dolphin.utils as utils
import dolphin.io_utils as io_utils
from dolphin.models import MODELS
import dolphin.preprocess.feature_extraction as feature_extraction



def generate_features(data, savedir, cfg):
    feature_type = cfg["preprocess"]["features"]

    # Note: The number of seconds in the loaded wav file is data.shape[0] / sr
    fp = data.name
    data, sr = librosa.load(data, sr=cfg['preprocess']['sampling_rate'])

    # Chunk the potentially long audio file into 3sec windows
    three_second_wavs = chunk(data, sr)

    # Generate features (ex. spectrograms) for each 3sec window
    feature_fps, orig_fps = [], []
    for i,wav in enumerate(three_second_wavs):
        if feature_type == 'spec':
            feature, f, t = feature_extraction.compute_spectrogram(wav, sr=sr, cfg=cfg, random_pad=False)
        elif feature_type == 'melspec':
            feature = feature_extraction.compute_melspec(wav, sr=sr, cfg=cfg)
        elif feature_type == 'pcen':
            melspec = feature_extraction.compute_melspec(wav, sr=sr, cfg=cfg)
            feature = feature_extraction.compute_pcen(melspec, sr=sr, cfg=cfg)

        savename = savedir + fp[:-4] + '_' + str(i) + '.png'
        io_utils.save_fig(feature, f, t, output_dir=savename, cfg=cfg)

        feature_fps.append(savename)
        orig_fps.append(fp[:-3] + 'png')

    return feature_fps, orig_fps


def chunk(wav: np.ndarray, sr: int):
    """
    Given a filename, chunks it up into 3 second windows. Assumes 60k sample rate.

    Args:
        wav (np.ndarray): time series from librosa.load call
        sr (int): sampling rate

    Returns:
        (list): list of time series windows
    """
    if wav.shape[0] <= sr * 3:
        return [wav]
    else:
        chunks = []
        for i in range(0, len(wav), sr * 3):
            chunks.append(wav[i : i + (sr * 3)])
        return chunks


class InferenceDataGenerator(Sequence):
    """
    InferenceDataGenerator grabs and loads batches of data.
    """

    def __init__(self, feat_fps, orig_fps):

        self.names = []
        self.images = []   
        self.visual_purpose = [] 

        for i,fp in enumerate(feat_fps):
            img = cv2.imread(fp)
            datapoint = np.expand_dims(img / 255, axis=0)

            self.names.append(orig_fps[i])
            self.images.append(datapoint)
            self.visual_purpose.append(img)

        self.count = 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        self.count += 1
        return self.images[i], self.names[i]


def run(data, model_name, threshold, weights, cfg_filename="config.json"):

    with open(cfg_filename, "r") as f:
        cfg = json.load(f)

    # -----------------------------------------------------------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------------------------------------------------------
    inference_dir = 'outputs/ui/detection/spectrograms/'
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)
    feat_fps, orig_fps = generate_features(data, inference_dir, cfg)
    input_shape = utils.get_input_shape(inference_dir)

    # -----------------------------------------------------------------------------------------------------------------
    # Data Generators
    # -----------------------------------------------------------------------------------------------------------------
    classes = ['no-whistle', 'whistle']  # 0: no-whistle, 1: whistle
    n_classes = len(classes)
    inference_generator = InferenceDataGenerator(feat_fps, orig_fps)    

    # -----------------------------------------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------------------------------------
    model_json_path = 'weights/detector_model.json'
    model_json = open(model_json_path, 'r')
    loaded_model_json = model_json.read()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(weights)

    # -----------------------------------------------------------------------------------------------------------------
    # Get them predictions!
    # -----------------------------------------------------------------------------------------------------------------
    predictions = []
    confidences = []
    for img in inference_generator.images:
        output = model.predict(img).squeeze(0)  # get model prediction
        for o in output:
            if float(o) < threshold:
                predictions.append(0)
            else:
                predictions.append(1)
            confidences.append(format(o, '.2%'))
            
    return predictions, confidences, inference_generator.visual_purpose, inference_generator.names
        


if __name__ == "__main__":
    run()
