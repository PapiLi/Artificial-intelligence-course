import tensorflow as tf
import matplotlib.pyplot as plt
from Captcha_train import crack_captcha_cnn
from Captcha_train import MAX_CAPTCHA
from Captcha_train import CHAR_SET_LEN
from Captcha_data import gen_captcha_text_and_image
from Captcha_train import convert2gray
#from Captcha_train import keep_prob
import os
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
all_count=10

def captcha_test(all_count):
    output = crack_captcha_cnn()
    X_ = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
    keep_pro = tf.placeholder(tf.float32)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        print("Model restored.")
        count = 0

        for i in range(all_count):
            text, image = gen_captcha_text_and_image()
            gray_image = convert2gray(image)
            captcha_image = gray_image.flatten() / 255

            #归一化
            print("Model restored2.")
            text_list = sess.run(predict, feed_dict={X_: [captcha_image], keep_pro: 1})
            predict_text = text_list[0].tolist()
            predict_text = str(predict_text)
            predict_text = predict_text.replace("[", "").replace("]", "").replace(",", "").replace(" ", "")
            if text == predict_text:
                count += 1
                check_result = "，预测结果正确"
            else:
                f = plt.figure()
                ax = f.add_subplot(111)
                ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
                plt.imshow(image)
                plt.show()
                #check_result = "，预测结果不正确"
            print("正确: {}  预测: {}".format(text, predict_text)) #+ check_result)
        print("正确率:", count, "/", all_count)

#captcha_test(1)
