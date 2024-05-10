import os
import pytesseract
import pyautogui
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 定义屏幕区域
screen_region = (0, 0, 1920, 1080)

while True:
    # 截取屏幕区域
    screenshot = pyautogui.screenshot(region=screen_region)
    # 保存截图为文件
    screenshot_path = "/tmp/screenshot.png"
    screenshot.save(screenshot_path)

    model = tf.keras.models.load_model('/Users/yuii/Downloads/fruit-recognition-main/model.h5')
    img = mpimg.imread(screenshot_path)
    test_image = load_img(screenshot_path, target_size=(128, 128))
    test_image = np.expand_dims(test_image, axis=0)
    prediction = model.predict(test_image)
    print(prediction)
    fruit_labels = ['苹果',  '香蕉','柠檬','芒果', '橙子']  # 定义一个列表，包含所有可能的水果类别
    predicted_class_index = np.argmax(prediction)  # 使用np.argmax函数找到预测结果中概率最大的类别的索引
    fruit_label = fruit_labels[predicted_class_index]  # 使用预测的类别索引从fruit_labels列表中获取对应的水果标签
    print("这是" + fruit_label )  # 打印出预测的水果标签
    plt.imshow(img)  # 使用imshow函数显示原始图像

    # 删除临时文件
    os.remove(screenshot_path)
