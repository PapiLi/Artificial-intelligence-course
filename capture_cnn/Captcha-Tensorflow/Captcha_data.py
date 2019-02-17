from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
from PIL import Image
import random

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


#从number列表的10个数中4次随机取数
def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


# 生成4个随机数字的验证码图片
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)  # 将数组转成字符串
    captcha = image.generate(captcha_text)
    # image.write(captcha_text, captcha_text + '.jpg')  # 写到文件

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image
