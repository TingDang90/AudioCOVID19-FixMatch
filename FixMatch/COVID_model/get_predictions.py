# -*- coding: utf-8 -*-
"""
@author: T. Quinnell, based on code from T. Xia
"""
from __future__ import print_function

import argparse
import os
import random
import sys
import joblib

import nni
import numpy as np
import tensorflow as tf
import tf_slim as slim
from tfdeterminism import patch

patch()
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
tf.set_random_seed(SEED)

NNI = False  # used for phper-parameter search


import model_params as params  # noqa: E402
import model_util as util  # noqa: E402
from model_network import *  # noqa: E402

sys.path.append("../vggish")
import warnings  # noqa: E402

from vggish_slim import load_vggish_slim_checkpoint  # noqa: E402

warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--train_name", type=str, default=params.AUDIO_TRAIN_NAME, help="Name of this programe.")
parser.add_argument("--task", type=str, default=params.TASK, help="Name of the task.")
parser.add_argument("--data_name", type=str, default=params.DATA_NAME, help="Original data path.")
parser.add_argument("--unlabelled_data_name", type=str, default=params.UNLABELLED_DATA_NAME, help="Original unlabelled data path.")
parser.add_argument("--is_aug", type=bool, default=False, help="Add data augmentation.")
parser.add_argument("--restore_if_possible", type=bool, default=False, help="Restore variables.")
parser.add_argument("--modality", type=str, default=params.MODALITY, help="Breath, cough, or voice.")
parser.add_argument("--reg_l2", type=float, default=params.L2, help="L2 regulation.")
parser.add_argument("--lr_decay", type=float, default=params.LEARNING_RATE_DECAY, help="learning rate decay rate.")
parser.add_argument("--dropout_rate", type=float, default=params.DROPOUT_RATE, help="Dropout rate.")
parser.add_argument("--epoch", type=int, default=20, help="Maximum epoch to train.")
parser.add_argument(
    "--early_stop", type=str, default=params.EARLY_STOP, help="The indicator on validation set to stop training."
)
parser.add_argument("--modfuse", type=str, default=params.MODFUSE, help="The method to fusing modalities.")
parser.add_argument("--is_diff", type=bool, default=False, help="Whether to use differential learing rate.")
parser.add_argument("--train_vgg", type=bool, default=False, help="Fine tuning Vgg")
parser.add_argument(
    "--trained_layers", type=int, default=params.TRAINED_LAYERS, help="The number Vgg layers to be fine tuned."
)
parser.add_argument("--rnn_units", type=int, default=32, help="The numer of unit in rnn.")
parser.add_argument("--num_units", type=int, default=64, help="The numer of unit in network.")
parser.add_argument("--lr1", type=float, default=1e-4, help="learning rate for Vgg layers.")
parser.add_argument("--lr2", type=float, default=1e-4, help="learning rate for top layers.")
parser.add_argument("--loss_weight", type=float, default=1, help="loss weight for symptom prediction.")
parser.add_argument("--is_sym", type=bool, default=False, help="Use symptom prediction.")
parser.add_argument("--disable_checkpoint", type=bool, default=False, help="Use pretrained VGGosh.")
parser.add_argument("--pseudo_label_threshold", type=float, default=0.9, help="Threshold for pseudo label training cutoff.")
parser.add_argument("--unlabelled_loss_weight", type=float, default=0.33)
FLAGS, _ = parser.parse_known_args()
# FLAGS = flags.FLAGS
tuner_params = nni.get_next_parameter()
FLAGS = vars(FLAGS)
FLAGS.update(tuner_params)

THRESHOLD = FLAGS['pseudo_label_threshold']
print("here is the threshold:", THRESHOLD)

data_dir = os.path.join(params.TF_DATA_DIR, FLAGS["data_name"])  # ./data
unlabelled_data_dir = os.path.join(params.TF_DATA_DIR, FLAGS["unlabelled_data_name"])
tensorboard_dir = os.path.join(params.TENSORBOARD_DIR, FLAGS["train_name"])  # ./data/tensorbord/
audio_ckpt_dir = os.path.join(
    params.AUDIO_CHECKPOINT_DIR, FLAGS["train_name"]
)  # ./data/train/ name_modality: name, with/out feature, modality: B, C, V, BCV
name_pre = (
    FLAGS["modality"]
    + "_"
    + "Dp"
    + str(FLAGS["dropout_rate"])
    + "_"
    + "U"
    + str(FLAGS["num_units"])
    + "_"
    + "R"
    + str(FLAGS["rnn_units"])
)
name_mid = "DC" + str(FLAGS["lr_decay"]) + "_" + "LR" + str(FLAGS["lr1"]) + "_" + str(FLAGS["lr2"])
name_pos = "MF" + str(FLAGS["modfuse"]) + "_" + "Aug" + str(FLAGS["is_aug"])
name_all = name_pre + "__" + name_mid + "__" + name_pos + "__"
print("save:", name_all)

util.maybe_create_directory(tensorboard_dir)
util.maybe_create_directory(audio_ckpt_dir)


def _create_data():
    """Create audio `train`, `test` and `val` records file."""
    tf.logging.info("Create records..")
    train, val, test = util.load_data(data_dir, FLAGS["is_aug"], shuffle_samples=False)
    tf.logging.info("Dataset size: Train-{} Test-{} Val-{}".format(len(train), len(test), len(val)))
    return train, val, test


def _create_unlabelled_data():
    """Create audio `train` records file for unlabelled data."""
    tf.logging.info("Create records for unlabelled data..")
    unlabelled_samples = util.load_unlabelled_data(unlabelled_data_dir)
    tf.logging.info("Dataset size: Train-{}".format(len(unlabelled_samples)))
    return unlabelled_samples


def _check_vggish_ckpt_exists():
    """check VGGish checkpoint exists or not."""
    util.maybe_create_directory(params.VGGISH_CHECKPOINT_DIR)
    if not util.is_exists(params.VGGISH_CHECKPOINT):
        url = "https://storage.googleapis.com/audioset/vggish_model.ckpt"
        util.maybe_download(url, params.VGGISH_CHECKPOINT_DIR)


def main(_):
    # initialize all log data containers:
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    # test_loss_per_epoch = []
    if FLAGS["early_stop"] == "LOSS":
        val_best = 100  # loss
    elif FLAGS["early_stop"] == "AUC":
        val_best = 0  # AUC
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(graph=add_student_teacher_graph(FLAGS, reuse_model=True), config=sess_config) as sess:
        # create tensors from session graph tensors
        tensors_student = TensorHolder(sess, params.VGGISH_INPUT_TENSOR_NAME, model_name="student/")
        
        summary_writer = tf.summary.FileWriter(tensorboard_dir, graph=sess.graph)
        student_saver = tf.train.Saver({util.remove_scope_name(tf_var.name, "student").split(":")[0]: tf_var for tf_var in util.remove_variables_in_scope(util.find_variables_in_scope("student/"), "student/train")})

        init = tf.global_variables_initializer()
        sess.run(init)
        if not FLAGS["disable_checkpoint"]:
            _check_vggish_ckpt_exists()
            load_vggish_slim_checkpoint(sess, params.VGGISH_CHECKPOINT, model_scope_name="student/")

        print(FLAGS)
        model_summary()

        checkpoint_path = os.path.join(audio_ckpt_dir, name_all + params.AUDIO_CHECKPOINT_NAME)
        if util.is_exists(checkpoint_path + ".meta"):
            student_saver.restore(sess, checkpoint_path)

        # load data
        unlabelled_data = _create_unlabelled_data()
        train_data, valid_data, test_data = _create_data() 

        logfile = open(os.path.join(audio_ckpt_dir, name_all + "_log.txt"), "w")
        logfile.write("INIT testing results:")
        logfile.write("\n")

        # prediction loop
            
        print("--------------------------------------")
        predictions_labelled, predictions_unlabelled, predictions_test = [], [], []
        print("labelled training samples:", len(train_data))
        for sample in train_data + valid_data:  # generate training batch
            vggcomb, index, index2, labels, symptom = util.get_input(sample)
            # Get predictions
            [logits, loss, latent] = sess.run(
                tensors_student.prediction_tensors(),
                feed_dict=tensors_student.feed_dict(vgg_tensor=vggcomb, index_tensor=index, index2_tensor=index2,
                    dropout_tensor=[[1.0]], symptom_tensor=symptom, labels_tensor=[labels]
                ),
            )
            predictions_labelled.append({"logits": logits, "loss": loss, "latent": latent, "labels": labels, "sym": "asym" if "no" in sample["sym"] else "sym", "info": sample["info"]})
            #predictions_labelled.append({"logits": logits, "loss": loss, "latent": latent, "labels": labels, "info": sample["info"]})

        print("unlabelled training samples: ", len(unlabelled_data))
        for sample in unlabelled_data:
            vggcomb, index, index2, labels, symptom = util.get_input(sample)
            # Get prediction data
            [logits, loss, latent] = sess.run(
                tensors_student.prediction_tensors(),
                feed_dict=tensors_student.feed_dict(vgg_tensor=vggcomb, index_tensor=index, index2_tensor=index2,
                    dropout_tensor=[[1.0]], symptom_tensor=symptom, labels_tensor=[labels]
                ),
            )
            predictions_unlabelled.append({"logits": logits, "loss": loss, "latent": latent, "labels": labels})
        print("test samples: ", len(test_data))  
        for sample in test_data:
            vggcomb, index, index2, labels, symptom = util.get_input(sample)
            [logits, loss, latent] = sess.run(
                tensors_student.prediction_tensors(),
                feed_dict=tensors_student.feed_dict(vgg_tensor=vggcomb, index_tensor=index, index2_tensor=index2,
                    dropout_tensor=[[1.0]], symptom_tensor=symptom, labels_tensor=[labels]
                ),
            )
            predictions_test.append({"logits": logits, "loss": loss, "latent": latent, "labels": labels, "sym": "asym" if "no" in sample["sym"] else "sym", "info": sample["info"]})
            #predictions_test.append({"logits": logits, "loss": loss, "latent": latent, "labels": labels, "info": sample["info"]})

        prediction_total = {"labelled": predictions_labelled, "unlabelled": predictions_unlabelled, "test": predictions_test}
       
        if not os.path.exists("./predictions"):
            os.mkdir("./predictions")
        with open("./predictions/pred_" + FLAGS["train_name"] + ".pk", "wb") as f:
            joblib.dump(prediction_total, f)
 

if __name__ == "__main__":
    tf.app.run()
