# author: T.XIA & T.Quinnell

"""Defines the 'audio' model used to classify the VGGish features."""

from __future__ import print_function

import sys

import model_params as params
import tensorflow as tf
import tf_slim as slim

sys.path.append("../vggish")
import vggish_slim  # noqa: E402

import warnings
warnings.filterwarnings("ignore")  # noqa: E402


def define_audio_slim(
    modality=params.MODALITY,
    reg_l2=params.L2,
    rnn_units=32,
    num_units=params.NUM_UNITS,
    modfuse=params.MODFUSE,
    train_vgg=False,
):
    """Defines the audio TensorFlow model.

    All ops are created in the current default graph, under the scope 'audio/'.

    The input is a placeholder named 'audio/vggish_input' of type float32 and
    shape [batch_size, feature_size] where batch_size is variable and
    feature_size is constant, and feature_size represents a VGGish output feature.
    The output is an op named 'audio/prediction' which produces the activations of
    a NUM_CLASSES layer.

    Args:
        training: If true, all parameters are marked trainable.

    Returns:
        The op 'mymodel/logits'.
    """

    embeddings = vggish_slim.define_vggish_slim(train_vgg)  # (? x 128) vggish is the pre-trained model
    print("model summary:", train_vgg)

    with slim.arg_scope(
        [slim.fully_connected],
        weights_initializer=tf.truncated_normal_initializer(
            stddev=params.INIT_STDDEV, seed=0
        ),  # 1 is the best for old data
        biases_initializer=tf.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(reg_l2),
    ), tf.variable_scope("mymodel"):

        index = tf.placeholder(dtype=tf.int32, shape=(1, 1), name="index")  # split B C V
        index2 = tf.placeholder(dtype=tf.int32, shape=(1, 1), name="index2")
        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_rate")

        if "B" in modality:
            with tf.name_scope("Breath"):
                # breath branch
                fc_vgg_breath = embeddings[0 : index[0, 0], :]  # (len, 128)
                fc1_b = tf.reduce_mean(fc_vgg_breath, axis=0)
                fc2_b = tf.reshape(fc1_b, (-1, 128), name="vgg_b")

        if "C" in modality:
            with tf.name_scope("Cough"):
                # cough branch
                fc_vgg_cough = embeddings[index[0, 0] : index2[0, 0], :]
                fc1_c = tf.reduce_mean(fc_vgg_cough, axis=0)
                fc2_c = tf.reshape(fc1_c, (-1, 128), name="vgg_c")

        if "V" in modality:
            with tf.name_scope("Voice"):
                # voice branch
                fc_vgg_voice = embeddings[index2[0, 0] :, :]
                fc1_v = tf.reduce_mean(fc_vgg_voice, axis=0)
                fc2_v = tf.reshape(fc1_v, (-1, 128), name="vgg_v")

        with tf.name_scope("Output"):
            # fusing and classifier
            if modality == "BCV":  # combination of three modalities
                if modfuse == "concat":
                    fc3 = tf.concat((fc2_b, fc2_c, fc2_v), axis=1, name="vgg_comb")
                if modfuse == "add":
                    fc3 = tf.add(fc2_b, fc2_c, fc2_v, name="vgg_comb")
            if modality == "B":
                fc3 = fc2_b
            if modality == "C":
                fc3 = fc2_c
            if modality == "V":
                fc3 = fc2_v

            # classification
            fc3_dp = tf.nn.dropout(fc3, dropout_keep_prob[0, 0], seed=0)
            fc4 = slim.fully_connected(fc3_dp, num_units)
            fc4_dp = tf.nn.dropout(fc4, dropout_keep_prob[0, 0], seed=0)
            logits = slim.fully_connected(fc4_dp, params.NUM_CLASSES, activation_fn=None, scope="logits")
            tf.nn.softmax(logits, name="prediction")

        with tf.name_scope("symptom"):
            fc5_dp = tf.nn.dropout(fc3, dropout_keep_prob[0, 0], seed=0)
            fc5 = slim.fully_connected(fc5_dp, num_units)
            fc6_dp = tf.nn.dropout(fc5, dropout_keep_prob[0, 0], seed=0)
            logits_sym = slim.fully_connected(fc6_dp, params.NUM_SYMPTOMS, activation_fn=None, scope="logits_sym")
            tf.nn.sigmoid(logits_sym, name="prediction_sym")
        return logits, logits_sym


def model_summary():
    """Print model to log."""
    print("\n")
    print("=" * 30 + "Model Structure" + "=" * 30)
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    print("=" * 60 + "\n")


def add_training_graph(FLAGS):
    """Define the TensorFlow Graph. Moved to model_network.py for reusability."""
    with tf.Graph().as_default() as graph:
        logits, logits_sym = define_audio_slim(
            modality=FLAGS["modality"],
            reg_l2=FLAGS["reg_l2"],
            rnn_units=FLAGS["rnn_units"],
            num_units=FLAGS["num_units"],
            modfuse=FLAGS["modfuse"],
            train_vgg=FLAGS["train_vgg"],
        )
        tf.summary.histogram("logits", logits)
        # define training subgraph
        with tf.variable_scope("train"):
            labels = tf.placeholder(tf.float32, shape=[None, params.NUM_CLASSES], name="labels")
            symptoms = tf.placeholder(tf.float32, shape=[None, params.NUM_SYMPTOMS], name="symptoms")
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="cross_entropy")
            symptom_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=symptoms, logits=logits_sym)
            sym_loss = tf.reduce_mean(symptom_entropy, name="sym_loss")
            cla_loss = tf.reduce_mean(cross_entropy, name="cla_loss")
            reg_loss2 = tf.add_n(
                [tf.nn.l2_loss(v) * FLAGS["reg_l2"] for v in tf.trainable_variables() if "bias" not in v.name],
                name="reg_loss2",
            )

            if FLAGS["is_sym"]:
                loss = tf.add(tf.add(cla_loss, reg_loss2), FLAGS["loss_weight"] * sym_loss, name="loss_op")
                # loss = tf.add(reg_loss2, sym_loss, name='loss_op')
            else:
                loss = tf.add(reg_loss2, cla_loss, name="loss_op")

            tf.summary.scalar("loss", loss)
            # training
            global_step = tf.Variable(
                0,
                name="global_step",
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP],
            )

            # Use Learning Rate Decaying for top layers
            number_decay_steps = 3000 if FLAGS["is_aug"] else 1000  # approciately an epoch
            base_of = FLAGS["lr_decay"]
            lr1 = tf.train.exponential_decay(
                FLAGS["lr1"], global_step, number_decay_steps, base_of, staircase=True, name="train_lr1"
            )
            lr2 = tf.train.exponential_decay(
                FLAGS["lr2"], global_step, number_decay_steps, base_of, staircase=True, name="train_lr2"
            )

            if FLAGS["is_diff"]:  # use different learning rate for vgg and others
                print("--------------learning rate control-----------------")
                var1 = tf.trainable_variables()[0 : FLAGS["trained_layers"]]  # Vggish
                var2 = tf.trainable_variables()[18:]  # FCNs
                train_op1 = tf.train.AdamOptimizer(learning_rate=lr1, epsilon=params.ADAM_EPSILON).minimize(
                    loss, var_list=var1, global_step=global_step, name="train_op1"
                )
                train_op2 = tf.train.AdamOptimizer(learning_rate=lr2, epsilon=params.ADAM_EPSILON).minimize(
                    loss, var_list=var2, global_step=global_step, name="train_op2"
                )  # fixed 'var1'
                train_op = tf.group(train_op1, train_op2, name="train_op")  # noqa: F841
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=lr2, epsilon=params.ADAM_EPSILON)
                optimizer.minimize(loss, global_step=global_step, name="train_op")
        return graph


class TensorHolder:
    """ Holds a set of references to Tensors in the computation graph """
    def __init__(self, sess, vggish_input_tensor_name):
        """
        Parameters
        ----------
        sess : tf.Session
            The TensorFlow session object holding the Tensors
        vggish_input_tensor_name : str
            Name of the input Tensor to the VGGish model
        """
        self.vgg_tensor = sess.graph.get_tensor_by_name(vggish_input_tensor_name)
        self.index_tensor = sess.graph.get_tensor_by_name("mymodel/index:0")
        self.index2_tensor = sess.graph.get_tensor_by_name("mymodel/index2:0")
        self.dropout_tensor = sess.graph.get_tensor_by_name("mymodel/dropout_rate:0")
        self.logit_tensor = sess.graph.get_tensor_by_name("mymodel/Output/prediction:0")
        self.logitsym_tensor = sess.graph.get_tensor_by_name("mymodel/symptom/prediction_sym:0")
        self.symptom_tensor = sess.graph.get_tensor_by_name("train/symptoms:0")
        self.labels_tensor = sess.graph.get_tensor_by_name("train/labels:0")
        self.global_step_tensor = sess.graph.get_tensor_by_name("train/global_step:0")
        self.lr1_tensor = sess.graph.get_tensor_by_name("train/train_lr1:0")
        self.lr2_tensor = sess.graph.get_tensor_by_name("train/train_lr2:0")
        self.loss_tensor = sess.graph.get_tensor_by_name("train/loss_op:0")
        self.cla_loss_tensor = sess.graph.get_tensor_by_name("train/cla_loss:0")
        self.reg_loss_tensor = sess.graph.get_tensor_by_name("train/reg_loss2:0")
        self.sym_loss_tensor = sess.graph.get_tensor_by_name("train/sym_loss:0")
        self.train_op = sess.graph.get_operation_by_name("train/train_op")
        self.summary_op = tf.summary.merge_all()


    def training_tensors(self):
        """ Returns a the list of Tensors used in a training step """
        return [
            self.global_step_tensor,
            self.lr1_tensor,
            self.lr2_tensor,
            self.logit_tensor,
            self.logitsym_tensor,
            self.loss_tensor,
            self.summary_op,
            self.train_op,
            self.cla_loss_tensor,
            self.reg_loss_tensor,
            self.sym_loss_tensor,
        ]

    def vad_tensors(self):
        """ Returns a the list of Tensors used in a prediction step """
        return [self.logit_tensor, self.logitsym_tensor, self.loss_tensor, self.cla_loss_tensor, self.reg_loss_tensor]


    def feed_dict(self, vgg_tensor, index_tensor, index2_tensor, dropout_tensor, symptom_tensor, labels_tensor):
        """ Returns a dictionary of (Tensor, value) pairs to use in a training step """
        return {
            self.vgg_tensor: vgg_tensor,  # Mel-spetrogram
            self.index_tensor: index_tensor,  # breath, cough
            self.index2_tensor: index2_tensor,  # voice
            self.dropout_tensor: dropout_tensor,  # traning dropout rate
            self.symptom_tensor: symptom_tensor,
            self.labels_tensor: labels_tensor,
        }


def load_audio_slim_checkpoint(session, checkpoint_path):
    """Loads a pre-trained audio-compatible checkpoint.

    This function can be used as an initialization function (referred to as
    init_fn in TensorFlow documentation) which is called in a Session after
    initializating all variables. When used as an init_fn, this will load
    a pre-trained checkpoint that is compatible with the audio model
    definition. Only variables defined by audio will be loaded.

    Args:
        session: an active TensorFlow session.
        checkpoint_path: path to a file containing a checkpoint that is
          compatible with the audio model definition.
    """

    # Get the list of names of all audio variables that exist in
    # the checkpoint (i.e., all inference-mode audio variables).
    with tf.Graph().as_default():
        define_audio_slim(training=False)
        audio_var_names = [v.name for v in tf.global_variables()]

    # Get list of variables from exist graph which passed by session
    with session.graph.as_default():
        global_variables = tf.global_variables()

    # Get the list of all currently existing variables that match
    # the list of variable names we just computed.
    audio_vars = [v for v in global_variables if v.name in audio_var_names]

    # Use a Saver to restore just the variables selected above.
    saver = tf.train.Saver(audio_vars, name="audio_load_pretrained", write_version=1)
    saver.restore(session, checkpoint_path)
