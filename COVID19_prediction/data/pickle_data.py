# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 11:25:58 2021

@author: xiatong
"""

import os

import joblib
import librosa
import numpy as np
import pandas as pd

SR = 16000  # sample rate
SR_VGG = 16000  # VGG pretrained model sample rate
FRAME_LEN = int(SR / 10)  # 100 ms
HOP = int(FRAME_LEN / 2)  # 50%overlap, 5ms
MFCC_DIM = 13

path = "0426_EN_used_task2"  # Change to your data path
inputFile = "data_0426_en_task2.csv"
df = pd.read_csv(inputFile)

pos_sym = "covid" + "nosym"
neg_sym = "health" + "nosym"


df_train_pos = df[(df["label"] == 1) & (df["fold"] == "train") & (df["categs"] == pos_sym)]
df_train_neg = df[(df["label"] == 0) & (df["fold"] == "train") & (df["categs"] == neg_sym)]
df_vad_pos = df[(df["label"] == 1) & (df["fold"] == "validation") & (df["categs"] == pos_sym)]
df_vad_neg = df[(df["label"] == 0) & (df["fold"] == "validation") & (df["categs"] == neg_sym)]
df_test_pos = df[(df["label"] == 1) & (df["fold"] == "test") & (df["categs"] == pos_sym)]
df_test_neg = df[(df["label"] == 0) & (df["fold"] == "test") & (df["categs"] == neg_sym)]

user_all = {}
user_all["train_covid_id"] = set([u for u in df_train_pos["uid"]])
user_all["train_noncovid_id"] = set([u for u in df_train_neg["uid"]])
user_all["vad_covid_id"] = set([u for u in df_vad_pos["uid"]])
user_all["vad_noncovid_id"] = set([u for u in df_vad_neg["uid"]])
user_all["test_covid_id"] = set([u for u in df_test_pos["uid"]])
user_all["test_noncovid_id"] = set([u for u in df_test_neg["uid"]])

opensmile_df = pd.read_csv("features_384.csv")


def get_feature(file):
    y, sr = librosa.load(file, sr=SR, mono=True, offset=0.0, duration=None)
    yt, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
    yt_n = yt / np.max(np.abs(yt))  # normolized the sound
    return yt_n


def get_opensmile(uid, folder_name=None):
    if folder_name:
        data_row = opensmile_df[(opensmile_df['uid'] == uid) & (opensmile_df['folder_name'] == folder_name)]
    else:
        data_row = opensmile_df[opensmile_df['uid'] == uid]
    df_cough = data_row["cough_feature"].map(lambda x: [float(v) for v in x.split(";")])
    cough = np.array([x for x in df_cough])
    df_breath = data_row["breath_feature"].map(lambda x: [float(v) for v in x.split(";")])
    breath = np.array([x for x in df_breath])
    df_voice = data_row["voice_feature"].map(lambda x: [float(v) for v in x.split(";")])
    voice = np.array([x for x in df_voice])
    return (breath, cough, voice)


def get_android(uid, COVID):
    folds = uid
    data = []
    fold_path = os.listdir(os.path.join(path, folds))
    for files in fold_path:  # date, no more than 5, check label
        samples = os.listdir(os.path.join(path, folds, files))
        print(files)
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
        b_opensmile, c_opensmile, v_opensmile = get_opensmile(uid, files)
        data.append({"breath": breath, "cough": cough, "voice": voice, "label": COVID, "breath_opensmile": b_opensmile, "cough_opensmile": c_opensmile, "voice_opensmile": v_opensmile, "info": os.path.join(path, folds, files)})
    return data


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
    b_opensmile, c_opensmile, v_opensmile = get_opensmile(uid)
    data.append({"breath": breath, "cough": cough, "voice": voice, "label": COVID, "breath_opensmile": b_opensmile, "cough_opensmile": c_opensmile, "voice_opensmile": v_opensmile, "info": os.path.join(path, folds, uid)})
    return data

# create directory to store Pickled data if it doesn't exist
if not os.path.exists("./data"):
    os.mkdir("./data")

# pickle postive users
COVID = 1
data_all_covid = {}  #
for fold in ["train_covid_id", "vad_covid_id", "test_covid_id"]:
    data_all_covid[fold] = {}
    for uid in user_all[fold]:
        print("==", uid, "===")
        if "202" in uid:
            temp = get_web(uid, COVID)
            if len(temp) > 0:
                data_all_covid[fold][uid] = temp
        else:
            temp = get_android(uid, COVID)
            if len(temp) > 0:
                data_all_covid[fold][uid] = temp
print("Saving in", "./data/audio_0426En_" + pos_sym + ".pk")

f = open("./data/audio_0426En_" + pos_sym + ".pk", "wb")
joblib.dump(data_all_covid, f)
f.close()

for fold in data_all_covid:
    print(fold, len(data_all_covid[fold]))
del data_all_covid


# pickle negative users
COVID = 0
data_all_noncovid = {}
for fold in ["train_noncovid_id", "vad_noncovid_id", "test_noncovid_id"]:
    data_all_noncovid[fold] = {}
    for uid in user_all[fold]:
        print("==", uid, "===")
        if "202" in uid:
            temp = get_web(uid, COVID)
            if len(temp) > 0:
                data_all_noncovid[fold][uid] = temp
        else:
            temp = get_android(uid, COVID)
            if len(temp) > 0:
                data_all_noncovid[fold][uid] = temp
print("Saving in", "./data/audio_0426En_" + neg_sym + ".pk")
for fold in data_all_noncovid:
    print(fold, len(data_all_noncovid[fold]))


f = open("./data/audio_0426En_" + neg_sym + ".pk", "wb")
joblib.dump(data_all_noncovid, f)
f.close()

