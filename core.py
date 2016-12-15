from __future__ import division
import operator
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

    img_resized = cv2.resize(img, (int(width), int(height)))
    return img_resized


def resize_image_string(img_str, width=None, height=None):
    file_bytes = np.asarray(bytearray(img_str), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    img_resized_string = resize_image(img, width, height)
    return convert_image(img_resized_string)


def crop_over(img1, img2, point1, point2):
    """
    Puts specific region of img2 onto img1 by taking transparency into account
    Paramaters
    ----------
    img1: 2-D ndarray
    img2: 2-D ndarray
    point1: tuple, two element, x and y coordinates of pivot point of img1
    point2: tuple, two element, x and y coordinates of pivot point of img2

    Return
    ----------
    cropped_image: 2-D ndarray, cropped image according to boundaries of img1
    region: tuple, four element, roi indices
    """
    p1x, p1y = point1
    p2x, p2y = point2

    h1, w1 = img1.shape[0], img1.shape[1]

    # translation parameters
    tx = p1x - p2x
    ty = p1y - p2y

    # translate and apply conditions
    grid = np.indices(img2.shape[0:2])

    ny = grid[0] + ty
    nx = grid[1] + tx

    my = (ny > 0) & (ny < h1)
    mx = (nx > 0) & (nx < w1)

    rows, cols = np.where(mx & my)
    r1 = rows[0]
    r2 = rows[-1]
    c1 = cols[0]
    c2 = cols[-1]

    cropped_img2 = img2[r1:r2, c1:c2].copy()

    # update point on img2
    p2y_new = p2y - r1
    p2x_new = p2x - c1

    # define roi
    roi_x1 = int(p1x - p2x_new)
    roi_y1 = int(p1y - p2y_new)

    roi_x2 = roi_x1 + cropped_img2.shape[1]
    roi_y2 = roi_y1 + cropped_img2.shape[0]

    region = (roi_x1, roi_y1, roi_x2, roi_y2)
    return cropped_img2, region


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
    # Get roi
    x1, y1, x2, y2 = region
    roi = img1[y1:y2, x1:x2]

    # select layer for binary thresholding
    img1_channel_no = img1.shape[-1]
    img2_channel_no = img2.shape[-1]

    if img2_channel_no == 3:
        img2_one = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    elif img2_channel_no == 4:
        img2_one = np.copy(img2[:, :, -1])
    else:
        img2_one = np.copy(img2[:, :, 0])

    # apply threshold, get masks
    ret, mask = cv2.threshold(img2_one, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg[:, :, :img1_channel_no])
    img1[y1:y2, x1:x2] = dst

    return img1


class Thing(object):
    def __init__(self, img_path, cx, cy, kind, scale_factor=None):
        self.image_path = img_path
        self.kind = kind
        self.cx = cx
        self.cy = cy
        if scale_factor is None:
            scale_factor = 1
        self.scale_factor = scale_factor

    @property
    def normalized_center(self):
        return self.cx, self.cy

    @property
    def image(self):
        return cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)


class Face(object):
    def __init__(self, img_path):
        self.image_path = img_path

        path, fileext = os.path.split(img_path)
        filename, ext = os.path.splitext(fileext)
        self.features_path = os.path.join(path, filename + '.json')

        self._features = None

    @property
    def image(self):
        return cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)

    @property
    def features(self):
        if self._features is None:
            result_data = utils.get_face_data(self.image_path, self.features_path, from_store=True)
            if result_data is not None:
                self._features = result_data[0]
        return self._features

    def get_glass_pivot(self):
        landmarks = self.features['faceLandmarks']
        if landmarks.get('eyebrowLeftOuter') and landmarks.get('eyebrowRightOuter'):
            nr_xc = int((landmarks['noseRootLeft']['x'] + landmarks['noseRootRight']['x']) / 2)
            nr_yc = int((landmarks['noseRootLeft']['y'] + landmarks['noseRootRight']['y']) / 2)

            eb_x1 = landmarks['eyebrowLeftOuter']['x']
            eb_x2 = landmarks['eyebrowRightOuter']['x']
            eb_w = int(abs(eb_x2 - eb_x1) * 1.2)

            return nr_xc, nr_yc, eb_w

    def get_hat_pivot(self):
        landmarks = self.features['faceLandmarks']

        p2x = int((landmarks['noseRootLeft']['x'] + landmarks['noseRootRight']['x']) / 2)
        p2y = int((landmarks['noseRootLeft']['y'] + landmarks['noseRootRight']['y']) / 2)

        ebo_left = np.array((landmarks['eyebrowLeftOuter']['x'], landmarks['eyebrowLeftOuter']['y']))
        ebo_right = np.array((landmarks['eyebrowRightOuter']['x'], landmarks['eyebrowRightOuter']['y']))

        new_width = np.linalg.norm(ebo_left - ebo_right) * 1.35

        return int(p2x), int(p2y), int(new_width)

    def wear_thing(self, img_face, thing, pivot):
        img_thing = thing.image
        p1x, p1y, new_width = pivot

        # prepare thing image
        cx, cy = thing.normalized_center

        # resize thing image
        img_thing = resize_image(img_thing, width=new_width * thing.scale_factor)
        gh, gw = img_thing.shape[0], img_thing.shape[1]
        p2x = int(gw * cx)
        p2y = int(gh * cy)

        # rotate thing image
        roll = 0
        try:
            roll = self.features['faceAttributes']['headPose']['roll']
        except:
            pass

        mmatrix = cv2.getRotationMatrix2D((p2x, p2y), -1 * roll, 1)
        img_thing = cv2.warpAffine(img_thing, mmatrix, (gw, gh))

        # crop thing image if necessary
        img_thing, region = crop_over(img_face, img_thing,
                                      (p1x, p1y), (p2x, p2y))

        # put thing image on face image
        img_face = put_onto(img_face, img_thing, region)

        return img_face

    def wear_accessories(self, accessories):
        img_face = self.image
        for ac in accessories:
            if ac.kind == 'glasses':
                pivot = self.get_glass_pivot()
            elif ac.kind == 'hat':
                pivot = self.get_hat_pivot()
            else:
                raise Exception('Undefined effect')

            img_face = self.wear_thing(img_face, ac, pivot)

        return img_face

    def mark_features(self):
        img_face = self.image

        for k, feature_point in self.features['faceLandmarks'].iteritems():
            if feature_point is not None:
                x, y = int(feature_point['x']), int(feature_point['y'])
                cv2.rectangle(img_face, (x, y), (x + 2, y + 2), (0, 255, 0), 2)
        return img_face

    # -------- lipstick effect --------
    #
    def get_mouth_roi(self):
        landmarks = self.features['faceLandmarks']
        delta = 2

        ml_x = landmarks['mouthLeft']['x'] - delta
        mr_x = landmarks['mouthRight']['x'] + delta

        uplt_y = landmarks['upperLipTop']['y'] - delta
        unlb_y = landmarks['underLipBottom']['y'] + delta

        x1 = int(ml_x)
        y1 = int(uplt_y)
        x2 = int(mr_x)
        y2 = int(unlb_y)

        return x1, y1, x2, y2

    #
    def get_lip_polygon(self):
        landmarks = self.features['faceLandmarks']

        p1x, p1y = landmarks['mouthLeft']['x'], landmarks['mouthLeft']['y']
        p2x, p2y = landmarks['upperLipTop']['x'], landmarks['upperLipTop']['y']
        p3x, p3y = landmarks['mouthRight']['x'], landmarks['mouthRight']['y']
        p4x, p4y = landmarks['underLipBottom']['x'], landmarks['underLipBottom']['y']

        pts = np.array([[p1x, p1y], [p2x, p2y], [p3x, p3y], [p4x, p4y + 3]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        return pts

    @staticmethod
    def poly2mask(img_shape, points):
        mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, points, 1)
        return mask

    @staticmethod
    def filter_by_range(img_channel, therange):
        low_end = therange[0]
        high_end = therange[1]

        lower_bound = np.array([low_end])
        upper_bound = np.array([high_end])
        mask = cv2.inRange(img_channel, lower_bound, upper_bound)
        return mask

    #
    @staticmethod
    def get_dominant_hues(img_hsv, mask=None):
        if mask is None:
            img_array = img_hsv.ravel()
        else:
            rows, cols = np.where(mask)
            img_array = img_hsv[rows, cols, 0].ravel()

        hist_norm, bin_edges = np.histogram(img_array, bins=181, range=(0, 181), density=True)
        hue_vals = np.where(hist_norm > 0.03)[0]
        return hue_vals

    @staticmethod
    def get_dominant_saturations(img_hsv, mask=None):
        val_range = (0, 256)
        val_max = 256
        channel = 1

        if mask is None:
            img_array = img_hsv.ravel()
        else:
            rows, cols = np.where(mask)
            img_array = img_hsv[rows, cols, channel].ravel()

        hist_norm, bin_edges = np.histogram(img_array, bins=val_max, range=val_range, density=True)
        sat_vals = np.where(hist_norm > 0.005)[0]
        return sat_vals

    def get_lip_mask_by_hue(self, img_mouth_hsv, hue_vals):
        if len(hue_vals) == 0:
            return np.ones(img_mouth_hsv.shape[0:2], dtype=np.uint8)

        masks = [self.filter_by_range(img_mouth_hsv[:, :, 0], (hv - 0.5, hv + 0.5)) for hv in hue_vals]
        lip_mask = reduce(operator.add, masks)
        return lip_mask

    def get_lip_mask_by_saturation(self, img_mouth_hsv, sat_vals):
        if len(sat_vals) == 0:
            return np.ones(img_mouth_hsv.shape[0:2], dtype=np.uint8)

        lip_mask = self.filter_by_range(img_mouth_hsv[:, :, 1], (sat_vals.min(), sat_vals.max()))
        return lip_mask

    @staticmethod
    def paint_by_mask(img_hsv, mask, clr_rgb):
        rows, cols = np.where(mask)
        clr_hsv = cv2.cvtColor(np.uint8([[clr_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        img_hsv[rows, cols, 0:2] = clr_hsv[0:2]

        clr_val_channel = clr_hsv[-1]
        mixed_values = [int(clr_val_channel * 0.4 + v * 0.6) for v in img_hsv[rows, cols, 2]]
        img_hsv[rows, cols, 2] = mixed_values
        return img_hsv

    def wear_lipstick(self, lipstick_rgb):
        img_face_bgr = self.image
        img_face_hsv = cv2.cvtColor(img_face_bgr, cv2.COLOR_BGR2HSV)

        # crop mouth region
        x1, y1, x2, y2 = self.get_mouth_roi()
        img_mouth_hsv = np.copy(img_face_hsv[y1:y2, x1:x2, :])

        # get lips polygon mask
        points = self.get_lip_polygon()
        lip_mask_polygon = self.poly2mask(img_face_hsv.shape, points)

        # derive mask for lips
        hue_vals = self.get_dominant_hues(img_face_hsv, lip_mask_polygon)
        lip_mask_hue = self.get_lip_mask_by_hue(img_mouth_hsv, hue_vals)

        sat_vals = self.get_dominant_saturations(img_face_hsv, lip_mask_polygon)
        lip_mask_sat = self.get_lip_mask_by_saturation(img_mouth_hsv, sat_vals)

        lip_mask_shape = lip_mask_polygon[y1:y2, x1:x2]

        # lip_mask_final = lip_mask_hue * lip_mask_sat
        lip_mask_final = np.logical_or(lip_mask_hue * lip_mask_sat, lip_mask_shape).astype(np.uint8)

        # apply painting on lips
        img_face_hsv[y1:y2, x1:x2] = self.paint_by_mask(img_mouth_hsv, lip_mask_final, lipstick_rgb)

        return cv2.cvtColor(img_face_hsv, cv2.COLOR_HSV2BGR)
