# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time
import glob
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/sample/"
count_fake = 0
count_total = 0

# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def test(image_name, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    # image_cropper = CropImage()
    image = cv2.imread(image_name)
    # result = check_image(image)
    # if result is False:
    #     return
    # image_bbox = model_test.get_bbox(image)
    # prediction = np.zeros((1, 2))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        # h_input, w_input, model_type, scale = parse_model_name(model_name)
        # param = {
        #     "org_img": image,
        #     "bbox": image_bbox,
        #     "scale": scale,
        #     "out_w": w_input,
        #     "out_h": h_input,
        #     "crop": True,
        # }
        # if scale is None:
        #     param["crop"] = False
        # img = image_cropper.crop(**param)
        start = time.time()
        prediction = model_test.predict(image, os.path.join(model_dir, model_name))
        print(prediction)
        test_speed += time.time()-start
        break

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]
    if label == 1:
        global count_fake
        count_fake += 1
    global count_total
    count_total += 1
    print(label, value)
    # if label == 1:
    #     print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
    #     result_text = "RealFace Score: {:.2f}".format(value)
    #     color = (255, 0, 0)
    # else:
    #     print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
    #     result_text = "FakeFace Score: {:.2f}".format(value)
    #     color = (0, 0, 255)
    # print("Prediction cost {:.2f} s".format(test_speed))
    # cv2.rectangle(
    #     image,
    #     (image_bbox[0], image_bbox[1]),
    #     (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
    #     color, 2)
    # cv2.putText(
    #     image,
    #     result_text,
    #     (image_bbox[0], image_bbox[1] - 5),
    #     cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

    # format_ = os.path.splitext(image_name)[-1]
    # result_image_name = image_name.replace(format_, "_result" + format_)
    # cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_folder",
        type=str,
        default="./images",
        help="folder containing images used to test")
    args = parser.parse_args()

    # Get list of all image files in the folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']  # Add more extensions if needed
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_folder, ext)))

    # Loop through each image and call the test function
    for image_path in image_paths:
        test(image_path, args.model_dir, args.device_id)
    print(f"đoán là fake: {count_fake}/{count_total}")
