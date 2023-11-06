# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:18:28 2020

@author: T. Quinnell, T. Xia
"""
from __future__ import print_function

import random

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow as tf

SR = 16000  # sample rate
import os  # noqa: E402
import sys  # noqa: E402

import librosa  # noqa: E402
import librosa.display  # noqa: E402
import model_params as params  # noqa: E402

sys.path.append("../vggish")

from vggish_input import waveform_to_examples  # noqa: E402

SR_VGG = params.SR_VGG

import warnings  # noqa: E402
warnings.filterwarnings("ignore")


def get_resort(files):
    """Re-sort the files under data path.

    :param files: file list
    :type files: list
    :return: alphabetic orders
    :rtype: list
    """
    name_dict = {}
    for sample in files:
        name = sample.lower()
        name_dict[name] = sample
    re_file = [name_dict[s] for s in sorted(name_dict.keys())]
    np.random.seed(222)
    np.random.shuffle(re_file)
    return re_file


def get_resort_test(files):
    """Re-sort the files under data path.

    :param files: file list
    :type files: list
    :return: alphabetic orders
    :rtype: list
    """
    name_dict = {}
    for sample in files:
        name = sample.lower()
        name_dict[name] = sample
    re_file = [name_dict[s] for s in sorted(name_dict.keys())]

    return re_file


def get_aug(y, type):
    """Augment data for training, validation and testing.
    :param data_path: path
    :type data_path: str
    :param is_aug: using augmentation
    :type is_aug: bool
    :return: batch
    :rtype: list
    """
    if type == "noise":
        y_aug = y + 0.005 * np.random.normal(0, 1, len(y))
    if type == "pitchspeed":
        step = np.random.uniform(-6, 6)
        y_aug = librosa.effects.pitch_shift(y, SR, step)
    yt_n = y_aug / np.max(np.abs(y_aug))  # re-normolized the sound
    return yt_n


def display_spec(spec, filename="spec_vis.png"):
    """ display a spectrogram, saving it to a file"""
    spec = spec.copy()
    if len(spec.shape) == 3:
        spec = spec.reshape((spec.shape[0] * spec.shape[1], spec.shape[2]))
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(spec, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()


def augment_weakly(spec, warp_min=0.95, warp_max=0.99, num_mask=1, freq_mask=0.05, time_mask=0.05):
    """ augmemt a spectrogram weakly """
    return spec_augment(spec, time_warp_rate=np.random.uniform(warp_min, warp_max), num_mask=num_mask, freq_masking_max_percentage=freq_mask, time_masking_max_percentage=time_mask)


def augment_strongly(spec, warp_min=0.8, warp_max=0.98, num_mask=2, freq_mask=0.35, time_mask=0.50):
    """ augment a spectrogram strongly """
    return spec_augment(spec, time_warp_rate=np.random.uniform(warp_min, warp_max), num_mask=num_mask, freq_masking_max_percentage=freq_mask, time_masking_max_percentage=time_mask)


def spec_augment(spec: np.ndarray, time_warp_rate=0.99, num_mask=2, freq_masking_max_percentage=0.1, time_masking_max_percentage=0.2):
    """ SpecAugment algorithm. Based on original code, edited to augment spectrograms seperately
    Parameters
    ----------
    spec : np.ndarray
        spectrogram to augment. This is not edited as a deep copy is taken.
    time_warp_rate : float
        the rate to warp. Currently unimplemented.
    num_mask : int
        the number of masks to apply to the spectrogram
    freq_masking_max_percentage : float
        maximum portion of frequency dimension to mask.
        Range: [0, 1]
    time_masking_max_percentage : float
        maximum portion of time dimension to mask.
        Range: [0, 1]
    """
    spec = spec.copy()
    spec_shape = spec.shape
    # for each sample and mask
    for spec_sample in spec:
        for i in range(num_mask):
            # mask frequency dimension
            all_frames_num, all_freqs_num = spec.shape[1:]
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)

            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec_sample[:, f0 : f0 + num_freqs_to_mask] = 0

            # mask time dimension
            time_percentage = random.uniform(0.0, time_masking_max_percentage)

            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec_sample[t0 : t0 + num_frames_to_mask, :] = 0
    spec = spec.reshape(spec_shape)
    return spec


def upsample(pos_dataset, neg_dataset, upsample_num=2):
    """ Upsamples/repeats the datasets to balance samples 
    Parameters
    ----------
    pos_dataset : list
        list of positive samples
    neg_dataset : list
        list of negative samples
    upsample_num : int
        number of times to upsample
    """
    if upsample_num:
        # upsampling by repeating some samples to balance the classes
        np.random.seed(1)
        covidcnt, noncvcnt = len(pos_dataset), len(neg_dataset)
        num_new_samples = upsample_num * abs(covidcnt - noncvcnt)
        if covidcnt < noncvcnt:
            add_covid = np.random.choice(range(covidcnt), num_new_samples, replace=num_new_samples>covidcnt)
            add_sample = [pos_dataset[i] for i in add_covid]
            pos_dataset = pos_dataset + add_sample
        else:
            add_noncovid = np.random.choice(range(noncvcnt), num_new_samples, replace=num_new_samples>noncvcnt)
            add_sample = [neg_dataset[i] for i in add_noncovid]
            neg_dataset = neg_dataset + add_sample
        print("added covid samples:", num_new_samples, "result:", len(pos_dataset), len(neg_dataset))
    return pos_dataset, neg_dataset


def load_data(data_path, is_aug, train_data_portion=1, shuffle_samples=True):
    """ Load data for training, validation and testing.
    Updated to limit train data portion
    :param data_path: path
    :type data_path: str
    :param is_aug: using augmentation
    :type is_aug: bool
    :param train_data_portion: portion of labelled data
    :type train_data_portion: int
    :param shuffle_samples: whether to shuffle samples in datasets
    :type shuffle_samples: boolean
    :return: batch
    :rtype: list
    """
    print("start to load data:", data_path)
    data = joblib.load(open(data_path + "_covid.pk", "rb"))  # load positive samples
    data2 = joblib.load(open(data_path + "_noncovid.pk", "rb"))  # load negative samples
    data.update(data2)

    train_task_pos, train_task_neg = [], []
    covidcnt = 0
    noncvcnt = 0
    for uid in get_resort(data["train_covid_id"]):
        for temp in data["train_covid_id"][uid]:
            train_task_pos.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [0, 1], "sym": temp["sym"], "info": temp["info"]}
            )
            covidcnt += 1
    for uid in get_resort(data["train_noncovid_id"]):
        for temp in data["train_noncovid_id"][uid]:
            train_task_neg.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [1, 0], "sym": temp["sym"], "info": temp["info"]}
            )
            noncvcnt += 1
    print("covid:", covidcnt, "non-covid:", noncvcnt)

    # upsampling
    train_task_pos, train_task_neg = upsample(train_task_pos, train_task_neg, upsample_num=None)

    # Reduce the training dataset size
    if train_data_portion != 1:
        print("Reducing training data size")
        np.random.seed(222)
        np.random.shuffle(train_task_pos)
        np.random.seed(222)
        np.random.shuffle(train_task_neg)
        train_task_pos = train_task_pos[:int(len(train_task_pos) * train_data_portion)]
        train_task_neg = train_task_neg[:int(len(train_task_neg) * train_data_portion)]
        covidcnt, noncvcnt = len(train_task_pos), len(train_task_neg)
        print("Dataset size is now", "covid:", covidcnt, "non-covid:", noncvcnt)

    train_task = train_task_pos + train_task_neg
    total = len(train_task)
    print("Length is", total)

    if is_aug:  # only works for train
        for i, type in enumerate(["_augnoise.pk", "_augpitch.pk"]):  #
            data_aug = joblib.load(open(data_path + type, "rb"))
            aug_covid = data_aug["covid"]
            aug_noncovid = data_aug["noncovid"]
            np.random.seed(i + 2)  # random and different
            add_covid = np.random.choice(range(covidcnt), (noncvcnt - covidcnt), replace=False)
            add_sample = [aug_covid[i] for i in add_covid]
            train_task = train_task + aug_covid + add_sample + aug_noncovid

    vad_task_pos, vad_task_neg = [], []
    covidcnt = 0
    noncvcnt = 0
    for uid in get_resort(data["vad_covid_id"]):
        for temp in data["vad_covid_id"][uid]:
            vad_task_pos.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [0, 1], "sym": temp["sym"], "info": temp["info"]}
            )
            covidcnt += 1
    for uid in get_resort(data["vad_noncovid_id"]):
        for temp in data["vad_noncovid_id"][uid]:
            vad_task_neg.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [1, 0], "sym": temp["sym"], "info": temp["info"]}
            )
            noncvcnt += 1
    vad_task = vad_task_pos + vad_task_neg
    print("covid:", len(vad_task_pos), "non-covid:", len(vad_task_neg))

    test_task = []
    covidcnt = 0
    noncvcnt = 0
    for uid in get_resort(data["test_covid_id"]):
        for temp in data["test_covid_id"][uid]:
            test_task.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [0, 1], "sym": temp["sym"], "info": temp["info"]}
            )
            covidcnt += 1
    for uid in get_resort(data["test_noncovid_id"]):
        for temp in data["test_noncovid_id"][uid]:
            test_task.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [1, 0], "sym": temp["sym"], "info": temp["info"]}
            )
            noncvcnt += 1
    print("covid:", covidcnt, "non-covid:", noncvcnt)

    # shuffle samples
    if shuffle_samples:
        np.random.seed(222)
        np.random.shuffle(train_task)
        np.random.seed(222)
        np.random.shuffle(vad_task)
        np.random.seed(222)
        np.random.shuffle(test_task)

    return train_task, vad_task, test_task


def load_vad_data(data_path, train_data_portion=1):
    """ Load vad data shuffled into train data """
    print("start to load data:", data_path)
    data = joblib.load(open(data_path + "_covid.pk", "rb"))
    data2 = joblib.load(open(data_path + "_noncovid.pk", "rb"))
    data.update(data2)

    vad_task_pos, vad_task_neg = [], []
    covidcnt = 0
    noncvcnt = 0

    # i = 0
    for uid in get_resort(data["train_covid_id"]):
        for temp in data["train_covid_id"][uid]:
            vad_task_pos.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [0, 1]}
            )
            covidcnt += 1
    for uid in get_resort(data["train_noncovid_id"]):
        for temp in data["train_noncovid_id"][uid]:
            vad_task_neg.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [1, 0]}
            )
            noncvcnt += 1
    for uid in get_resort(data["vad_covid_id"]):
        for temp in data["vad_covid_id"][uid]:
            vad_task_pos.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [0, 1]}
            )
            covidcnt += 1
    for uid in get_resort(data["vad_noncovid_id"]):
        for temp in data["vad_noncovid_id"][uid]:
            vad_task_neg.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [1, 0]}
            )
            noncvcnt += 1

    print("covid:", covidcnt, "non-covid:", noncvcnt)
    vad_task_pos, vad_task_neg = upsample(vad_task_pos, vad_task_neg, upsample_num=None)

    # Reduce the training dataset size
    if train_data_portion != 1:
        print("Reducing training data size")
        np.random.seed(222)
        np.random.shuffle(vad_task_pos)
        np.random.seed(222)
        np.random.shuffle(vad_task_neg)
        vad_task_pos = vad_task_pos[:int(len(vad_task_pos) * train_data_portion)]
        vad_task_neg = vad_task_neg[:int(len(vad_task_neg) * train_data_portion)]
        covidcnt, noncvcnt = len(vad_task_pos), len(vad_task_neg)
        print("Dataset size is now", "covid:", covidcnt, "non-covid:", noncvcnt)

    vad_task = vad_task_pos + vad_task_neg
    total = len(vad_task)
    print("Length is", total)

    print("covid:", covidcnt, "non-covid:", noncvcnt)
    np.random.seed(222)
    np.random.shuffle(vad_task)
    return vad_task


def load_test_data(data_path):
    """Load test data only."""
    print("start to load data:", data_path)
    data = joblib.load(open(data_path + "_covid.pk", "rb"))
    data2 = joblib.load(open(data_path + "_noncovid.pk", "rb"))
    data.update(data2)

    test_task = []
    covidcnt = 0
    noncvcnt = 0

    i = 0
    for uid in get_resort_test(data["test_covid_id"]):
        for temp in data["test_covid_id"][uid]:
            i += 1
            test_task.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [0, 1]}
            )
            covidcnt += 1
    for uid in get_resort_test(data["test_noncovid_id"]):
        for temp in data["test_noncovid_id"][uid]:
            i += 1
            test_task.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [1, 0]}
            )
            noncvcnt += 1
    return test_task


def count_samples(sample_dict):
    """ count the number of samples across the sample dictionary
    Parameters:
    ----------
    sample_dict : dict
        a dictionary of the format (Fold: List<Samples>)
    """
    count = 0
    for k in sample_dict.keys():
        count += len(sample_dict[k])
    return count


def merge_unlabelled_samples(dict1, dict2):
    """ Merge dictionaries of unlabelled samples
    Parameters:
    ----------
    dict1, dict2 : dict in form Fold -> User -> list of samples
        dictionaries holding the samples for both unlabelled data splits
    """
    for fold in dict1.keys():  # merge samples for folds in both
        if fold in dict2.keys():
            for user in dict2[fold].keys():
                dict1[fold][user] = dict2[fold][user]
    for fold in dict2.keys():  # copy any remaining folds
        if fold not in dict1.keys():
            dict1[fold] = dict2[fold]


def load_unlabelled_data(data_path):
    """ Load unlabelled data only.
    Parameters
    ----------
    data_path : str
        the filepath to load the unlabelled data from
    """
    print("start to load data:", data_path)
    data = joblib.load(open(data_path + "1.pk", "rb"))
    data2 = joblib.load(open(data_path + "2.pk", "rb"))
    merge_unlabelled_samples(data, data2)

    unlabelled_task = []
    covidcnt = 0
    noncvcnt = 0

    # Get unlabelled data samples, label fixed for simplicity
    for uid in get_resort_test(data["test_covid_id"]):
        for temp in data["test_covid_id"][uid]:
            unlabelled_task.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [0, 1] if temp["label"] == 1 else [1, 0]}
            )
            covidcnt += 1
    print(f"Loaded {covidcnt} samples for unlabelled data.")
    return unlabelled_task


def shuffle_unlabelled(train_data, unlabelled_data, val_data=None, data_path=None, train_data_portion=1):
    """ Shuffle unlabelled data into train data
    Parameters
    ----------
    train_data : list
        list of labelled train data samples
    unlabelled_data : list
        list of unlabelled data samples
    val_data : list
        list of validation data samples
        if not None, the data is shuffled into training data
    data_path : str
        filepath to load data path from
    train_data_portion : float
        portion of training data
    """
    # first tag datasets - labelled and unlabelled
    train_data_tagged = [(sample, "labelled") for sample in train_data]
    if val_data:
        train_data_tagged = [(sample, "labelled") for sample in load_vad_data(data_path, train_data_portion=train_data_portion)]
    unlabelled_data_tagged = [(sample, "unlabelled") for sample in unlabelled_data]
    
    # now join and shuffle the datasets
    joint_dataset = train_data_tagged
    for sample in unlabelled_data_tagged:
        joint_dataset.append(sample)
    np.random.seed(222)
    np.random.shuffle(joint_dataset)
    return joint_dataset


def merge_spec(sample1, sample2, alpha=0.2):
    """ merge 2 spectrograms for use in MixUp
        merge samples from the same  modality together
    Parameters
    sample1, sample2 : Tuple
        samples to merge.
        In format: spectrogram, index1, index2, label, symptom_tensor
    """
    # first balance out number of samples in each spectrogram
    spec1, spec2 = sample1[0], sample2[0]
    spec1_shape, spec2_shape = spec1.shape, spec2.shape
    spec1_index1, spec1_index2 = [i[0][0] for i in sample1[1:3]]
    spec2_index1, spec2_index2 = [i[0][0] for i in sample2[1:3]]
    
    # repeat samples in each modality
    s1, s2 = [], []
    for [s1_s, s1_e], [s2_s, s2_e] in ([[[0, spec1_index1], [0, spec2_index1]], [[spec1_index1, spec1_index2], [spec2_index1, spec2_index2]], [[spec1_index2, spec1_shape[0]], [spec2_index2, spec2_shape[0]]]]):
        s1_range, s2_range = s1_e - s1_s, s2_e - s2_s
        for i in range(max(s1_range, s2_range)):
            s1.append(spec1[s1_s + i % s1_range])
            s2.append(spec2[s2_s + i % s2_range])
    s1, s2 = np.array(s1), np.array(s2)

    # now sample the Lambda parameter
    lam_value = np.random.beta(alpha, alpha)
    lam_value = max(lam_value, 1 - lam_value)

    # merge spectrograms and labels
    label1, label2 = np.array(sample1[-2]), np.array(sample2[-2])
    return (lam_value * s1 + (1 - lam_value) * s2), max(sample1[1], sample2[1]), max(sample1[2], sample2[2]), lam_value * label1 + (1 - lam_value) * label2, sample1[4]


def load_test_dict_data(data_path):
    """load dict data with labal."""
    print("start to load data:", data_path)
    data = joblib.load(open(data_path, "rb"))
    return data


def get_input_fixmatch(sample, strong_aug_mask=0.25, weak_aug_mask=0.05):
    """ transfer audio input into spectrogram, augmenting them for use in FixMatch
    Parameters:
    ----------
    sample : dict
        the sample object holding raw audio, label etc.
    strong_aug_mask : float
        maximum size for each mask in strong augmentation
    weak_aug_mask : float
        maxmimum size for each mask in weak augmentation
    """
    vgg_b = waveform_to_examples(sample["breath"], SR_VGG)
    vgg_c = waveform_to_examples(sample["cough"], SR_VGG)
    vgg_v = waveform_to_examples(sample["voice"], SR_VGG)

    index = vgg_b.shape[0]
    index2 = vgg_c.shape[0] + index

    labels = sample["label"]
    symptoms = [[1] * 13]  # sample['sym']

    # Weakly augment
    np.random.seed(222)
    vgg_b_w = augment_weakly(vgg_b, freq_mask=weak_aug_mask, time_mask=weak_aug_mask)
    vgg_c_w = augment_weakly(vgg_c, freq_mask=weak_aug_mask, time_mask=weak_aug_mask)
    vgg_v_w = augment_weakly(vgg_v, freq_mask=weak_aug_mask, time_mask=weak_aug_mask)
    vgg_input_w = np.concatenate((vgg_b_w, vgg_c_w, vgg_v_w), axis=0)

    # Strongly augment
    vgg_b_s = augment_strongly(vgg_b, freq_mask=strong_aug_mask, time_mask=strong_aug_mask)
    vgg_c_s = augment_strongly(vgg_c, freq_mask=strong_aug_mask, time_mask=strong_aug_mask)
    vgg_v_s = augment_strongly(vgg_v, freq_mask=strong_aug_mask, time_mask=strong_aug_mask)
    vgg_input_s = np.concatenate((vgg_b_s, vgg_c_s, vgg_v_s), axis=0)
    
    return vgg_input_w, vgg_input_s, [[index]], [[index2]], labels, symptoms


def get_fm_input_from_spec(sample, strong_aug_mask=0.15):
    """ transfer audio input into spectrogram, augmenting them for use in FixMatch.
        sample is a tuple of processed objects. Used in FixUp
    Parameters:
    ----------
    sample : Tuple
        the sample object holding raw audio, label etc.
    strong_aug_mask : float
        maximum size for each mask in strong augmentation
    weak_aug_mask : float
        maxmimum size for each mask in weak augmentation
    """
    spec, index1, index2 = sample[0: 3]
    index1 = index1[0][0]
    index2 = index2[0][0]
    vgg_b = spec[:index1]
    vgg_c = spec[index1: index2]
    vgg_v = spec[index2:]

    # Weakly augment
    np.random.seed(222)
    vgg_b_w = augment_weakly(vgg_b)
    vgg_c_w = augment_weakly(vgg_c)
    vgg_v_w = augment_weakly(vgg_v)
    vgg_input_w = np.concatenate((vgg_b_w, vgg_c_w, vgg_v_w), axis=0)

    # Strongly augment
    vgg_b_s = augment_strongly(vgg_b, freq_mask=strong_aug_mask, time_mask=strong_aug_mask)
    vgg_c_s = augment_strongly(vgg_c, freq_mask=strong_aug_mask, time_mask=strong_aug_mask)
    vgg_v_s = augment_strongly(vgg_v, freq_mask=strong_aug_mask, time_mask=strong_aug_mask)
    vgg_input_s = np.concatenate((vgg_b_s, vgg_c_s, vgg_v_s), axis=0)

    return (vgg_input_w, vgg_input_s, *sample[1:])


def get_input(sample, augment=False):
    """transfer audio input into spectrogram."""
    vgg_b = waveform_to_examples(sample["breath"], SR_VGG)
    vgg_c = waveform_to_examples(sample["cough"], SR_VGG)
    vgg_v = waveform_to_examples(sample["voice"], SR_VGG)
    if augment:
        vgg_b = augment_weakly(vgg_b)
        vgg_c = augment_weakly(vgg_c)
        vgg_v = augment_weakly(vgg_v)

    index = vgg_b.shape[0]
    index2 = vgg_c.shape[0] + index
    vgg_input = np.concatenate((vgg_b, vgg_c, vgg_v), axis=0)

    labels = convert_to_float(sample["label"])
    symptoms = [[1] * 13]  # sample['sym']
    return vgg_input, [[index]], [[index2]], labels, symptoms


def convert_to_float(l):
    """ convert a list of ints to floats """
    return [float(e) for e in l]


def get_metrics(probs, labels):
    """calculate metrics.
    :param probs: list
    :type probs: float
    :param labels: list
    :type labels: int
    :return: metrics
    """
    probs = np.array(probs)
    probs = np.squeeze(probs)

    predicted = []
    for i in range(len(probs)):
        if probs[i][0] > 0.5:
            predicted.append(0)
        else:
            predicted.append(1)

    label = np.array(labels)
    label = np.squeeze(label)

    predicted = np.array(predicted)
    predicted = np.squeeze(predicted)

    # pre = metrics.precision_score(label, predicted)
    # acc = metrics.accuracy_score(label, predicted)
    auc = metrics.roc_auc_score(label, probs[:, 1])
    precision, recall, _ = metrics.precision_recall_curve(label, probs[:, 1])
    # rec = metrics.recall_score(label, predicted)

    TN, FP, FN, TP = metrics.confusion_matrix(label, predicted).ravel()
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)

    # PPV = TP/(TP + FP)
    # NPV = TN/(TN + FN)

    fpr, tpr, thresholds = metrics.roc_curve(label, probs[:, 1])
    index = np.where(tpr > 0.9)[0][0] - 1
    print(
        "AUC:"
        + "{:.2f}".format(auc)
        + " Sensitivity:"
        + "{:.2f}".format(TPR)
        + " Specificity:"
        + "{:.2f}".format(TNR)
        + " spe@90%sen:"
        + "{:.2f}".format(1 - fpr[index])
    )

    return auc, TPR, TNR, 1 - fpr[index]


def get_metrics_t(probs, label):
    predicted = []
    for i in range(len(probs)):
        if probs[i] > 0.5:
            predicted.append(1)
        else:
            predicted.append(0)

    auc = metrics.roc_auc_score(label, probs)
    TN, FP, FN, TP = metrics.confusion_matrix(label, predicted).ravel()
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP * 1.0 / (TP + FN)
    # Specificity or true negative rate
    TNR = TN * 1.0 / (TN + FP)

    return auc, TPR, TNR


def get_CI(data, AUC, Sen, Spe):
    AUCs = []
    TPRs = []
    TNRs = []
    for s in range(1000):
        np.random.seed(s)  # Para2
        sample = np.random.choice(range(len(data)), len(data), replace=True)
        samples = [data[i] for i in sample]
        sample_pro = [x[0] for x in samples]
        sample_label = [x[1] for x in samples]
        try:
            get_metrics_t(sample_pro, sample_label)
        except ValueError:
            np.random.seed(1001)  # Para2
            sample = np.random.choice(range(len(data)), len(data), replace=True)
            samples = [data[i] for i in sample]
            sample_pro = [x[0] for x in samples]
            sample_label = [x[1] for x in samples]
        else:
            auc, TPR, TNR = get_metrics_t(sample_pro, sample_label)
        AUCs.append(auc)
        TPRs.append(TPR)
        TNRs.append(TNR)

    q_0 = pd.DataFrame(np.array(AUCs)).quantile(0.025)[0]  # 2.5% percentile
    q_1 = pd.DataFrame(np.array(AUCs)).quantile(0.975)[0]  # 97.5% percentile

    q_2 = pd.DataFrame(np.array(TPRs)).quantile(0.025)[0]  # 2.5% percentile
    q_3 = pd.DataFrame(np.array(TPRs)).quantile(0.975)[0]  # 97.5% percentile

    q_4 = pd.DataFrame(np.array(TNRs)).quantile(0.025)[0]  # 2.5% percentile
    q_5 = pd.DataFrame(np.array(TNRs)).quantile(0.975)[0]  # 97.5% percentile

    print(
        str(AUC.round(2))
        + "("
        + str(q_0.round(2))
        + "-"
        + str(q_1.round(2))
        + ")"
        + "&"
        + str(Sen.round(2))
        + "("
        + str(q_2.round(2))
        + "-"
        + str(q_3.round(2))
        + ")"
        "&" + str(Spe.round(2)) + "(" + str(q_4.round(2)) + "-" + str(q_5.round(2)) + ")"
    )


def is_exists(path):
    """Check directory exists."""
    if not os.path.exists(path):
        print("Not exists: {}".format(path))
        return False
    return True


def maybe_create_directory(dirname):
    """Check directory exists or create it."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def maybe_download(url, dirname):
    """Check resource exists or download it"""
    resource_name = url.split("/")[-1]
    resource_dest = os.path.join(dirname, resource_name)
    if not os.path.isfile(resource_dest):
        r = requests.get(url)
        with open(resource_dest, "wb") as f:
            f.write(r.content)


# model - already trained keras model with dropout
def mc_dropout(predictions, T):
    # predictions shape: (I, T, C) T - monte carlo samples, I input size, C number of classes

    # shape: (I, C)
    mean = np.mean(predictions, axis=1)
    mean = np.squeeze(mean)
    print("mean:", mean.shape)

    # shape: (I)
    variance = -1 * np.sum(np.log(mean) * mean, axis=1)
    return (mean, variance)


def find_variable(var_name):
    """ Lookup the TF Variable Object named as var_name """
    global_vars = tf.global_variables()
    for global_var in global_vars:
        if global_var.name == var_name:
            return global_var
    return None


def find_variables_in_scope(scope_name):
    """ Lookup the TF Variable Objects which start with scope_name """
    vars_in_scope = []
    for tf_var in tf.global_variables():
        if tf_var.name.startswith(scope_name):
            vars_in_scope.append(tf_var)
    return vars_in_scope


def remove_scope_name(var_name, scope_name):
    """ Remove instances of scope_name in var_name """
    new_name = []
    for scope_part in var_name.split("/"):
        if scope_part != scope_name:
            new_name.append(scope_part)
    return "/".join(new_name)


def remove_variables_in_scope(variables, scope_name):
    """ Remove TF Variables from variables in scope scope_name"""
    return list(filter(lambda v: not v.name.startswith(scope_name), variables))


def plot_prob_hist(probs, filename):
    """ Plot probaility histogram """
    plt.hist(probs, 100, density=True)
    plt.savefig(filename)
    plt.clf()


def confidence_histogram(probs, labels):
    """ plot histograms of confidences from probabilities """
    # make directory to save in
    save_dir = os.path.join(os.getcwd(), "histograms")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # split into pos and neg probs
    # predicted_actual_probabilities
    all_pos_pred = []
    pos_pos_pred_probs = []
    neg_pos_pred_probs = []
    pos_neg_pred_probs = []
    neg_neg_pred_probs = []
    for prob, label in zip(probs, labels):
        prob = prob[0]
        all_pos_pred.append(prob[1])
        if label == 1:
            pos_pos_pred_probs.append(prob[1])
            neg_pos_pred_probs.append(prob[0])
        else:
            pos_neg_pred_probs.append(prob[1])
            neg_neg_pred_probs.append(prob[0])
    # histogram of positive probs
    plot_prob_hist(all_pos_pred, os.path.join(save_dir, "all_pos_pred.png"))
    plot_prob_hist(pos_pos_pred_probs, os.path.join(save_dir, "act_pos_pred_pos_hist.png"))
    plot_prob_hist(neg_pos_pred_probs, os.path.join(save_dir, "act_neg_pred_pos_hist.png"))
    plot_prob_hist(pos_neg_pred_probs, os.path.join(save_dir, "act_pos_pred_neg_hist.png"))
    plot_prob_hist(neg_neg_pred_probs, os.path.join(save_dir, "act_neg_pred_neg_hist.png"))

