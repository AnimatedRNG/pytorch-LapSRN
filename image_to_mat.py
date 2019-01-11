import argparse
import numpy as np
import cv2
from scipy.io import savemat


def convert_to_mat(img):
    down = cv2.resize(img, (0, 0),
                      fx=1/4, fy=1/4,
                      interpolation=cv2.INTER_CUBIC)
    up = cv2.resize(down, (0, 0),
                    fx=4, fy=4,
                    interpolation=cv2.INTER_CUBIC)
    mat = {
        'im_gt_y': img,
        'im_b_y': up,
        'im_l_y': down,
    }

    return mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image")
    parser.add_argument("output_image")
    args = parser.parse_args()
    try:
        with open(args.input_image, 'rb') as f:
            img_buffer = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(
                img_buffer, cv2.IMREAD_GRAYSCALE)
            img_center = (img.shape[0] // 2, img.shape[1] // 2)
            mat = convert_to_mat(img[img_center[0] - 200: img_center[0] + 200,
                                     img_center[1] - 200: img_center[1] + 200])
            savemat(args.output_image, mat)
    except EnvironmentError:
        print("Unable to open {}".format(args.input_image))
