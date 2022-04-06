import os
import sys
import cv2
import json
import math
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from io import StringIO
from tensorflow.keras import optimizers
from tensorflow.keras.utils import Sequence, to_categorical

import dolphin.utils as utils
import dolphin.io_utils as io_utils
from dolphin.models import MODELS
import dolphin.preprocess.feature_extraction as feature_extraction


def generate_features(data_list, fns_to_times, savedir, cfg):
    feature_type = cfg["preprocess"]["features"]
    spec_max_length = cfg["preprocess"]["spectrogram_max_length"]

    for data in data_list:
        fp = data.name
        data, sr = librosa.load(data, sr=cfg['preprocess']['sampling_rate'])

        for i,time in enumerate(fns_to_times[fp[:-4]]):
            start = math.floor(float(time[0]) * sr)
            dur = int(spec_max_length) * sr
            chunk = data[start : start + dur]
            
            feature, f, t = feature_extraction.compute_spectrogram(chunk, sr=sr, cfg=cfg, random_pad=False)
            savename = savedir + fp[:-4] + str(i) + '.png'
            io_utils.save_fig(feature, f, t, output_dir=savename, cfg=cfg)

    return savedir


class InferenceDataGenerator(Sequence):
    """
    InferenceDataGenerator grabs and loads batches of data.
    """

    def __init__(self, infer_dir, wav_files, fns_to_times):

        self.names = []       
        self.indices = []
        self.images = []   
        self.visual_purpose = []

        for wav in wav_files:
            basename = wav.name[:-4]
            n_chunks = len(fns_to_times[basename])

            for n in range(n_chunks):
                fp = infer_dir + basename + str(n) + '.png'

                img = cv2.imread(fp)
                datapoint = np.expand_dims(img / 255, axis=0)
                self.names.append(basename)
                self.indices.append(n)
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
    fns_to_times = {}
    wav_files = []
    dfs = []

    for data in uploaded_data:
        basename = data.name[:-4]

        # If the list of chunks for this file hasn't yet been defined, initialize it
        if not basename in fns_to_times:
            fns_to_times[basename] = []

        # Read in the annotations file, extracting the start/stop time of each detected whistle
        if data.name.endswith('.csv'):
            df = pd.read_csv(data)
            dfs.append(df)
            for i,row in df.iterrows():
                fns_to_times[basename].append([row['Begin Time (s)'], row['End Time (s)']])

        # Append the data (UploadedData type) to the list of whistles that need to be loaded in
        if data.name.endswith('.wav'):
            wav_files.append(data)     

    # -----------------------------------------------------------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------------------------------------------------------
    inference_dir = 'outputs/ui/raven_classification/spectrograms/'  # for the purpose of running spectrograms through inference
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)

    generate_features(wav_files, fns_to_times, inference_dir, cfg)
    input_shape = utils.get_input_shape(inference_dir)

    # -----------------------------------------------------------------------------------------------------------------
    # Data Generators
    # -----------------------------------------------------------------------------------------------------------------
    classes = np.sort(['INSERT_CLASS1', 'INSERT_CLASS2', 'INSERT_CLASS3'])
    n_classes = len(classes)
    inference_generator = InferenceDataGenerator(inference_dir, wav_files, fns_to_times)    

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

        confidence = [format(output[ind[2]], '.2%'), format(output[ind[1]], '.2%'), format(output[ind[0]], '.2%')]  # confidence scores of top 2 predictions  
        prediction = [classes[ind[2]], classes[ind[1]], classes[ind[0]]]  # extract the actual classnames of these 2 predictions

        predictions.append(prediction)
        confidences.append(confidence)
    
    return predictions, confidences, inference_generator.visual_purpose, inference_generator.names, inference_generator.indices, dfs
        


if __name__ == "__main__":
    run()
