from ocr_tensorflow_cnn_freetype.genIDCard  import *
from ocr_tensorflow_cnn_freetype.tf_cnn_lstm_ctc  import *
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import cv2


def crack_image():
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step,
                                               DECAY_STEPS,
                                               LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    logits, inputs, targets, seq_len, W, b = get_train_model()

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, "./ocr.model-6800")
        # test_inputs,test_targets,test_seq_len = get_next_batch(1)
        test_inputs, test_targets, test_seq_len, image = get_a_image()
        test_feed = {inputs: test_inputs,
                     targets: test_targets,
                     seq_len: test_seq_len}
        dd, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)
        report_accuracy(dd, test_targets)
        plt.imshow(image)
        plt.show()
        # image.astype(np.uint8)
        print(type(image), image.shape, image.dtype)
        # image = image.astype(np.uint8)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)

if __name__ == '__main__':
    crack_image()