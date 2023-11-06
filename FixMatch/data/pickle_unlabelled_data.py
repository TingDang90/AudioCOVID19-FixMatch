# -*- coding: utf-8 -*-
"""
@author: xiatong, T. Quinnell
"""

import os

import joblib
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment

SR = 16000  # sample rate
SR_VGG = 16000  # VGG pretrained model sample rate
FRAME_LEN = int(SR / 10)  # 100 ms
HOP = int(FRAME_LEN / 2)  # 50%overlap, 5ms
MFCC_DIM = 13

path = "unlabelled_data"  # Change to your data path


def get_feature(file):
    if file.endswith(".wav"):
        y, sr = librosa.load(file, sr=SR, mono=True, offset=0.0, duration=None)
    elif file.endswith(".m4a"):
        y, sr = pydub_to_np(AudioSegment.from_file(file))
    yt, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
    yt_n = yt / np.max(np.abs(yt))  # normolized the sound
    return yt_n


def get_android(uid, COVID):
    folds = uid
    data = []
    fold_path = os.listdir(os.path.join(path, folds))
    for files in fold_path:  # date, no more than 5, check label
        samples = os.listdir(os.path.join(path, folds, files))
        if uid + "/" + files in unlabelled_labels["pos"]:
            COVID = 1
        elif uid + "/" + files in unlabelled_labels["neg"]:
            COVID = 0
        else:
            raise Exception
        
        for sample in samples:
            if "breath" in sample:
                file_b = os.path.join(path, folds, files, sample)
            if "cough" in sample:
                file_c = os.path.join(path, folds, files, sample)
            if "voice" in sample or "read" in sample:
                file_v = os.path.join(path, folds, files, sample)

        breath = get_feature(file_b)
        cough = get_feature(file_c)
        voice = get_feature(file_v)
        data.append({"breath": breath, "cough": cough, "voice": voice, "label": COVID})
    return data


def pydub_to_np(audio):
    """ Converts pydub AudioSegment into np array of shape (duration * sample_rate, channels) """
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)), audio.frame_rate


def get_web(uid, COVID):  # date
    folds = "form-app-users"
    data = []
    samples = os.listdir(os.path.join(path, folds, uid))
    for sample in samples:
        if "breath" in sample:
            file_b = os.path.join(path, folds, uid, sample)
        if "cough" in sample:
            file_c = os.path.join(path, folds, uid, sample)
        if "voice" in sample or "read" in sample:
            file_v = os.path.join(path, folds, uid, sample)

    breath = get_feature(file_b)
    cough = get_feature(file_c)
    voice = get_feature(file_v)
    data.append({"breath": breath, "cough": cough, "voice": voice, "label": COVID})
    return data

# create directory to store Pickled data if it doesn't exist
if not os.path.exists("./data"):
    os.mkdir("./data")

# split data into 2 by uid on split_char
split_char = "m"

# get labels
with open("./unlabelled_labels.pk", "rb") as f:
    unlabelled_labels = joblib.load(f)

# pickle postive users into test set to run predictions on
COVID = 1
data_all_covid = {}  #
count1 = 0
for fold in ["test_covid_id"]:
    data_all_covid[fold] = {}
    for uid in os.listdir(path):
        if uid.lower() <= split_char:
            print("==", uid, "===")
            if "202" in uid:
                print("WEB!!!")
                raise Exception
                temp = get_web(uid, COVID)
                if len(temp) > 0:
                    data_all_covid[fold][uid] = temp
            else:
                temp = get_android(uid, COVID)
                if len(temp) > 0:
                    data_all_covid[fold][uid] = temp
            count1 += len(temp)
print(count1, "samples in the first part")
f = open("./data/unlabelled_all_1.pk", "wb")
joblib.dump(data_all_covid, f)
f.close()
del data_all_covid

# split data into 2
COVID = 1
data_all_covid = {} 
count2 = 0
for fold in ["test_covid_id"]:
    data_all_covid[fold] = {}
    for uid in os.listdir(path):
        if uid.lower() > split_char:
            print("==", uid, "===")
            if "202" in uid:
                temp = get_web(uid, COVID)
                if len(temp) > 0:
                    data_all_covid[fold][uid] = temp
            else:
                temp = get_android(uid, COVID)
                if len(temp) > 0:
                    data_all_covid[fold][uid] = temp
            count2 += len(temp)
print(count2, "samples in the second part")
f = open("./data/unlabelled_all_2.pk", "wb")
joblib.dump(data_all_covid, f)
f.close()
print(count1 + count2, "samples overall")
