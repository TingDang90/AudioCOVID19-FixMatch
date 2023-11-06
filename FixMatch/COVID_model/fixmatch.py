# -*- coding: utf-8 -*-
"""
@author: T. Quinnell based on code from T. Xia
"""
from __future__ import print_function

import argparse
import os
import random
import sys

import nni
import numpy as np
import tensorflow as tf
import tf_slim as slim
from tfdeterminism import patch

patch()
SEED = 0
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
parser.add_argument("--pseudo_label_threshold", type=float, default=0.95, help="Threshold for pseudo label training cutoff.")
parser.add_argument("--unlabelled_loss_weight", type=float, default=0.33)
parser.add_argument("--train_data_portion", type=float, default=1, help="portion of labelled data to use when training.")
parser.add_argument("--shuffle_vad", type=str, default="y", help="whether to shuffle in validation data in last 5 epochs")
parser.add_argument("--dynamic_label", type=str, default="y", help="whether to dynamically psuedo-label")
FLAGS, _ = parser.parse_known_args()
# FLAGS = flags.FLAGS
tuner_params = nni.get_next_parameter()
FLAGS = vars(FLAGS)
FLAGS.update(tuner_params)
# parse string args into bool as parser cannot handle bools as expected
FLAGS["shuffle_vad"] = FLAGS["shuffle_vad"] == "y"
FLAGS["dynamic_label"] = FLAGS["dynamic_label"] == "y"

train_data_portion = FLAGS["train_data_portion"]
THRESHOLD = FLAGS['pseudo_label_threshold']
print("Psuedo-label threshold:", THRESHOLD)

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
    train, val, test = util.load_data(data_dir, FLAGS["is_aug"], train_data_portion=train_data_portion)
    tf.logging.info("Dataset size: Train-{} Test-{} Val-{}".format(len(train), len(test), len(val)))
    return train, val, test


def _create_unlabelled_data():
    """ Create audio `train` records for unlabelled data. """
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
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(graph=add_student_teacher_graph(FLAGS, reuse_model=FLAGS["dynamic_label"]), config=sess_config) as sess:
        # Load data
        unlabelled_data = _create_unlabelled_data()
        train_data, valid_data, test_data = _create_data()

        logfile = open(os.path.join(audio_ckpt_dir, name_all + "_log.txt"), "w")
        logfile.write("INIT testing results:")
        logfile.write("\n")

        # create tensors from session graph tensors
        tensors_student = TensorHolder(sess, params.VGGISH_INPUT_TENSOR_NAME, model_name="student/")
        tensors_teacher = TensorHolder(sess, params.VGGISH_INPUT_TENSOR_NAME, model_name=("student/" if FLAGS["dynamic_label"] else "teacher/"))
        
        summary_writer = tf.summary.FileWriter(tensorboard_dir, graph=sess.graph)
        student_saver = tf.train.Saver({util.remove_scope_name(tf_var.name, "student").split(":")[0]: tf_var for tf_var in util.remove_variables_in_scope(util.find_variables_in_scope("student/"), "student/train")})
        if not FLAGS["dynamic_label"]:
            teacher_saver = tf.train.Saver({util.remove_scope_name(tf_var.name, "teacher").split(":")[0]: tf_var for tf_var in util.remove_variables_in_scope(util.find_variables_in_scope("teacher/"), "teacher/train")})
        else:
            teacher_saver = student_saver

        # Iterate over strong augmentation levels
        for strong_aug_mask in range(5, 45, 10):
            strong_aug_mask /= 100
            print("Strong aug mask percent is now", strong_aug_mask)
            if FLAGS["early_stop"] == "LOSS":
                val_best = 100  # loss
            elif FLAGS["early_stop"] == "AUC":
                val_best = 0  # AUC
                sens_spec_best = 0
            init = tf.global_variables_initializer()
            sess.run(init)
            if not FLAGS["disable_checkpoint"]:
                _check_vggish_ckpt_exists()
                load_vggish_slim_checkpoint(sess, params.VGGISH_CHECKPOINT, model_scope_name="student/")
                if not FLAGS["dynamic_label"]:
                    load_vggish_slim_checkpoint(sess, params.VGGISH_CHECKPOINT, model_scope_name="teacher/")

            print(FLAGS)
            model_summary()

            # restore params
            checkpoint_path = os.path.join(
                                  os.path.join(params.AUDIO_CHECKPOINT_DIR, FLAGS["train_name"]),
                                  name_all + params.AUDIO_CHECKPOINT_NAME)
            print(checkpoint_path)
            if util.is_exists(checkpoint_path + ".meta"):
                student_saver.restore(sess, checkpoint_path)
                teacher_saver.restore(sess, checkpoint_path)
            checkpoint_path = os.path.join(audio_ckpt_dir + "_" + str(int(strong_aug_mask * 100)), name_all + params.AUDIO_CHECKPOINT_NAME)
            print(checkpoint_path)

            # shuffle together labelled and unlabelled data
            train_data_full = util.shuffle_unlabelled(train_data, unlabelled_data)

            # training and validation loop
            for epoch in range(20):
                if epoch == 0:
                    curr_step = 0
            
                print("--------------------------------------")
                # training loop
                train_batch_losses = []
                probs_all = []
                label_all = []
                probs_sym_all = []
                label_sym_all = []
                loss_all = []
                regloss_all = []
                symloss_all = []
                num_labelled, num_unlabelled = 0, 0

                # shuffle together labelled and unlabelled data
                print("training samples:", len(train_data_full))
                
                for sample, label_type in train_data_full:  # generate training batch
                    # train steps on labelled and unlabelled data 
                    if label_type == "labelled":
                        num_labelled += 1
                        vggcomb, index, index2, labels, symptom = util.get_input(sample)
                        # Training step with student network
                        [num_steps, lr1, lr2, logits, logitsym, loss, summaries, _, clal, regl, syml] = sess.run(
                            tensors_student.training_tensors(),
                            feed_dict=tensors_student.feed_dict(vgg_tensor=vggcomb, index_tensor=index, index2_tensor=index2,
                                dropout_tensor=[[FLAGS["dropout_rate"]]], symptom_tensor=symptom, labels_tensor=[labels]
                            ),
                        )
                    elif label_type == "unlabelled":
                        vggcomb_weak, vggcomb_strong, index, index2, labels, symptom = util.get_input_fixmatch(sample, strong_aug_mask=strong_aug_mask, weak_aug_mask=0.05)
                        # Prediction step with teacher network
                        [logits, logitsym, loss, clal, regl] = sess.run(
                            tensors_teacher.vad_tensors(),
                            feed_dict=tensors_teacher.feed_dict(vgg_tensor=vggcomb_weak, index_tensor=index, index2_tensor=index2,
                                dropout_tensor=[[1.0]], symptom_tensor=symptom, labels_tensor=[labels]
                            ),
                        )
                        
                        # check if above threshold
                        if np.max(logits) > THRESHOLD:
                            # convert to Onehot label
                            labels = logits[0]
                            max_index = np.argmax(labels)
                            labels = [1.0 if j == max_index else 0.0 for j in range(len(labels))]
                        else:
                            continue
                        # Training step with student network
                        num_unlabelled += 1
                        [num_steps, lr1, lr2, logits, logitsym, loss, summaries, _, clal, regl, syml] = sess.run(
                            tensors_student.training_tensors(),
                            feed_dict=tensors_student.feed_dict(vgg_tensor=vggcomb_strong, index_tensor=index, index2_tensor=index2,
                                dropout_tensor=[[FLAGS["dropout_rate"]]], symptom_tensor=symptom, labels_tensor=[labels], labelled_indicator=label_type=="unlabelled"
                            ),
                        )
                    else:
                        continue
                    
                    probs_all.append(logits)
                    label_all.append(labels[1])
                    probs_sym_all.append(logitsym)
                    label_sym_all.append(symptom)
                    train_batch_losses.append(loss)
                    loss_all.append(clal)
                    regloss_all.append(regl)
                    symloss_all.append(syml)
                    summary_writer.add_summary(summaries, num_steps)                  
                
                print("Number of labelled samples trained on this epoch:", num_labelled)
                print("Number of unlabelled samples trained on this epoch:", num_unlabelled)

                if FLAGS["is_diff"]:
                    print("LEARNING RATE1:", lr1, "Learning RATE2:", lr2)
                    print()
                else:
                    print("LEARNING RATE:", lr2)
            
                # compute the train epoch loss:
                train_epoch_loss = np.mean(train_batch_losses)
                epoch_loss = np.mean(loss_all)
                epcoh_reg_loss = np.mean(regloss_all)
                epcoh_sym_loss = np.mean(symloss_all)
                # save the train epoch loss:
                train_loss_per_epoch.append(train_epoch_loss)
                print(
                    "Epoch {}/{}:".format(epoch, FLAGS["epoch"]),
                    "train epoch loss: %g" % train_epoch_loss,
                    "cross-entropy loss: %g" % epoch_loss,
                    "regulation loss: %g" % epcoh_reg_loss,
                    "symptom loss: %g" % epcoh_sym_loss,
                )
                AUC, TPR, TNR, TPR_TNR_9 = util.get_metrics(probs_all, label_all)
                # if FLAGS["is_sym"]:
                #     UAR_SYM = util.get_metrics2(probs_sym_all, label_sym_all)
                # train_auc = AUC
                logfile.write(
                    "Epoch {}/{}: AUC:{}, TPR:{}, TNR:{}, TPR_TNR_9:{}".format(
                        epoch, FLAGS["epoch"], AUC, TPR, TNR, TPR_TNR_9
                    )
                )
                logfile.write("\n")
                
                # validation loop
                val_batch_losses = []
                probs_all = []
                label_all = []
                loss_all = []
                regloss_all = []
                probs_sym_all = []
                label_sym_all = []
                for sample in valid_data:
                #for sample in test_data:
                    vggcomb, index, index2, labels, symptom = util.get_input(sample)
                    [logits, logitsym, loss, clal, regl] = sess.run(
                        tensors_student.vad_tensors(),
                        feed_dict=tensors_student.feed_dict(vgg_tensor=vggcomb, index_tensor=index, index2_tensor=index2,
                            dropout_tensor=[[1.0]], symptom_tensor=symptom, labels_tensor=[labels]
                        ),
                    )

                    val_batch_losses.append(loss)
                    probs_all.append(logits)
                    label_all.append(labels[1])
                    loss_all.append(clal)
                    regloss_all.append(regl)
                    probs_sym_all.append(logitsym)
                    label_sym_all.append(symptom)
    
                val_loss = np.mean(val_batch_losses)
                val_loss_per_epoch.append(val_loss)
                epoch_loss = np.mean(loss_all)
                epcoh_reg_loss = np.mean(regloss_all)
                print(
                    "Epoch {}/{}:".format(epoch, FLAGS["epoch"]),
                    "validation loss: %g" % val_loss,
                    "cross-entropy loss: %g" % epoch_loss,
                    "regulation loss: %g" % epcoh_reg_loss,
                )
                AUC, TPR, TNR, TPR_TNR_9 = util.get_metrics(probs_all, label_all)
                # if FLAGS["is_sym"]:
                #     UAR_SYM = util.get_metrics2(probs_sym_all, label_sym_all)
                # vad_auc = AUC
                logfile.write(
                    "Epoch {}/{}: AUC:{}, TPR:{}, TNR:{}, TPR_TNR_9:{}".format(
                        epoch, FLAGS["epoch"], AUC, TPR, TNR, TPR_TNR_9
                    )
                )
                logfile.write("\n")
  
                # early stop
                if not FLAGS['shuffle_vad'] or epoch < 15:
                    if FLAGS["early_stop"] == "LOSS":
                        if val_loss <= val_best:
                            # save the model weights to disk:
                            student_saver.save(sess, checkpoint_path)
                            print("checkpoint saved in file: %s" % checkpoint_path)
                            curr_step = 0
                            val_best = val_loss
                        else:
                            curr_step += 1
                            if curr_step == params.PATIENCE:
                                print("Early Sopp!(Train)")
                                logfile.write("Min Val Loss, checkpoint stored!\n")
                                # break

                    elif FLAGS["early_stop"] == "AUC":
                        print("It is AUC", val_best, TPR, TNR)
                        if val_best <= 0.5 * (TPR + TNR) and TPR > 0.5 and TNR > 0.5:
                            # save the model weights to disk:
                            student_saver.save(sess, checkpoint_path)
                            print("checkpoint saved in file: %s" % checkpoint_path)
                            curr_step = 0
                            val_best = 0.5 * (TPR + TNR)
                        else:
                            curr_step += 1
                            if curr_step == params.PATIENCE:
                                print("Early Sopp!(Train)")
                                logfile.write("Max Val AUC, checkpoint stored!\n")
                                # break
                # test loop
                test_batch_losses = []
                probs_all = []
                label_all = []
                loss_all = []
                regloss_all = []
                probs_sym_all = []
                label_sym_all = []
                for sample in test_data:
                    vggcomb, index, index2, labels, symptom = util.get_input(sample)
                    [logits, logitsym, loss, clal, regl] = sess.run(
                        tensors_student.vad_tensors(),
                        feed_dict=tensors_student.feed_dict(vgg_tensor=vggcomb, index_tensor=index, index2_tensor=index2,
                            dropout_tensor=[[1.0]], symptom_tensor=symptom, labels_tensor=[labels]
                        ),
                    )

                    test_batch_losses.append(loss)
                    probs_all.append(logits)
                    label_all.append(labels[1])
                    loss_all.append(clal)
                    regloss_all.append(regl)
                    probs_sym_all.append(logitsym)
                    label_sym_all.append(symptom)

                test_loss = np.mean(test_batch_losses)
                epoch_loss = np.mean(loss_all)
                epcoh_reg_loss = np.mean(regloss_all)
                print(
                    "Epoch {}/{}:".format(epoch, FLAGS["epoch"]),
                    "test loss: %g" % test_loss,
                    "cross-entropy loss: %g" % epoch_loss,
                    "regulation loss: %g" % epcoh_reg_loss,
                )
                AUC, TPR, TNR, TPR_TNR_9 = util.get_metrics(probs_all, label_all)
                data = [[probs_all[i][0, 1], label_all[i]] for i in range(len(label_all))]
                util.get_CI(data, AUC, TPR, TNR)
                # if FLAGS["is_sym"]:
                #     UAR_SYM = util.get_metrics2(probs_sym_all, label_sym_all)
                # vad_auc = AUC
                logfile.write(
                    "Epoch {}/{}: AUC:{}, TPR:{}, TNR:{}, TPR_TNR_9:{}".format(
                        epoch, FLAGS["epoch"], AUC, TPR, TNR, TPR_TNR_9
                    )
                )
                logfile.write("\n")
                if FLAGS['shuffle_vad'] and epoch == 14:
                    print("start fine tune!")
                    train_data_full = util.shuffle_unlabelled(train_data, unlabelled_data, valid_data, data_path=data_dir, train_data_portion=train_data_portion)
                    try:
                        student_saver.restore(sess, checkpoint_path)
                        teacher_saver.restore(sess, checkpoint_path)
                        print(checkpoint_path)
                    except Exception as e:
                        print("Failed to load checkpoint:")
                        print(e)
                        checkpoint_path = os.path.join(audio_ckpt_dir + "_" + str(int(strong_aug_mask * 100)), name_all + params.AUDIO_CHECKPOINT_NAME)
                        print(checkpoint_path)
                        val_best = 0

            # Save at end
            if FLAGS['shuffle_vad']:
                student_saver.save(sess, checkpoint_path)
                print("checkpoint saved in file: %s" % checkpoint_path)

if __name__ == "__main__":
    tf.app.run()
