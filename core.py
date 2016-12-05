from __future__ import division
import os.path

import numpy as np
import cv2

import utils


def convert_image(img):
    """
    Paramaters
    ----------
    2-D ndarray

    Return
    ----------
    byte string
    """

    img_str = cv2.imencode('.jpg', img)[1].tostring()
    return img_str


def resize_image(img, width=None, height=None):
    """
    Resizes images

    Paramaters
    ----------
    2-D ndarray

    Return
    ----------
    2-D ndarray
    """

    if width is None and height is None:
        return img

    h, w = img.shape[:2]
    aspect_ratio = w / h

    if height is None:
        height = int(width / aspect_ratio)
    if width is None:
        width = int(height * aspect_ratio)

    img_resized = cv2.resize(img, (width, height))
    return img_resized


def resize_image_string(img_str, width=None, height=None):
    file_bytes = np.asarray(bytearray(img_str), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    img_resized_string = resize_image(img, width, height)
    return convert_image(img_resized_string)


def put_onto(img1, img2, region):
    """
    Puts specific region of img2 onto img1 by taking transparency into account

    Paramaters
    ----------
    img1: 2-D ndarray
    img2: 2-D ndarray
    region: tuple, four element, x and y coordinates of two points of region rectangle

    Return
    ----------
    2-D ndarray
    """

    img1_channel_no = img1.shape[-1]
    img2_channel_no = img2.shape[-1]

    if img2_channel_no == 3:
        img2_one = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    elif img2_channel_no == 4:
        img2_one = np.copy(img2[:, :, -1])
    else:
        img2_one = np.copy(img2[:, :, 0])

    ret, mask = cv2.threshold(img2_one, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_inv = cv2.bitwise_not(mask)

    # Get roi
    x1, y1, x2, y2 = region
    roi = img1[y1:y2, x1:x2]

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg[:, :, :img1_channel_no])
    img1[y1:y2, x1:x2] = dst

    return img1


class Glasses(object):
    def __init__(self, img_path):
        self.image_path = img_path

    def get_image(self):
        img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        return img

    def retouch_glass(self, img):
        # kernel = np.ones((2, 2), np.uint8)
        # img = cv2.dilate(img, kernel, iterations=1)
        # kernel = np.ones((3, 3), np.float32) / 9
        # dst = cv2.filter2D(img, -1, kernel)

        # kernel = np.ones((2, 2), np.uint8)
        # img = cv2.dilate(img, kernel, iterations=1)

        # img = cv2.blur(img, (5, 5))
        # img = cv2.GaussianBlur(img, (5, 5), 0)

        return img


class Face(object):
    def __init__(self, img_path):
        self.image_path = img_path

        path, fileext = os.path.split(img_path)
        filename, ext = os.path.splitext(fileext)
        self.features_path = os.path.join(path, filename + '.json')

        self._features = None

    @property
    def features(self):
        if self._features is None:
            result_data = utils.get_face_data(self.image_path, self.features_path, from_store=True)
            if result_data is not None:
                self._features = result_data[0]
        return self._features

    def get_image(self):
        img = cv2.imread(self.image_path)
        return img

    def get_glass_pivot(self, facefeatures):
        landmarks = facefeatures['faceLandmarks']
        if landmarks.get('eyebrowLeftOuter') and landmarks.get('eyebrowRightOuter'):
            nr_xc = int((landmarks['noseRootLeft']['x'] + landmarks['noseRootRight']['x']) / 2)
            eb_x1 = landmarks['eyebrowLeftOuter']['x'] * 0.95
            eb_x2 = landmarks['eyebrowRightOuter']['x'] * 1.05
            eb_xc = (eb_x1 + eb_x2) / 2

            gapx = eb_xc - nr_xc
            x1 = int(eb_x1 - gapx)
            x2 = int(eb_x2 - gapx)
            new_width = x2 - x1

            nr_yc = int((landmarks['noseRootLeft']['y'] + landmarks['noseRootRight']['y']) / 2)
            eb_y1 = int(landmarks['eyebrowLeftInner']['y'])
            eb_y2 = int(landmarks['eyebrowRightInner']['y'])
            eb_yc = (eb_y1 + eb_y2) / 2

            gapy = nr_yc - eb_yc
            y1 = int(eb_yc + 0.0 * gapy)
            return x1, y1, new_width

    def wear_glass(self, img_glass):
        img_face = self.get_image()

        roi_x1, roi_y1, new_width = self.get_glass_pivot(self.features)
        img_glass = resize_image(img_glass, width=new_width)

        roi_x2 = roi_x1 + img_glass.shape[1]
        roi_y2 = roi_y1 + img_glass.shape[0]
        region = (roi_x1, roi_y1, roi_x2, roi_y2)

        img_face = put_onto(img_face, img_glass, region)

        return img_face

    def mark_features(self):
        img_face = self.get_image()

        for k, feature_point in self.features['faceLandmarks'].iteritems():
            if feature_point is not None:
                x, y = int(feature_point['x']), int(feature_point['y'])
                cv2.rectangle(img_face, (x, y), (x + 2, y + 2), (0, 255, 0), 2)
        return img_face
