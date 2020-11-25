"""Perform data augmentation and preprocessing."""
import json
import math
import os
import random
import cv2 as cv
import numpy as np

def boundary_check(centralied_marks):
    """Check situation that marking point appears too near to border."""
    for mark in centralied_marks:
        if mark[0] < -260 or mark[0] > 260 or mark[1] < -260 or mark[1] > 260:
            return False
    return True

def overlap_check(centralied_marks):
    """Check situation that multiple marking points appear in same cell."""
    for i in range(len(centralied_marks) - 1):
        i_x = centralied_marks[i, 0]
        i_y = centralied_marks[i, 1]
        for j in range(i + 1, len(centralied_marks)):
            j_x = centralied_marks[j, 0]
            j_y = centralied_marks[j, 1]
            if abs(j_x - i_x) < 600 / 16 and abs(j_y - i_y) < 600 / 16:
                return False
    return True

def generalize_marks(centralied_marks, with_direction=False):
    """Convert coordinate to [0, 1] and calculate direction label."""
    generalized_marks = []
    for mark in centralied_marks:
        xval = (mark[0] + 300) / 600
        yval = (mark[1] + 300) / 600
        if with_direction:
            direction = math.atan2(mark[3] - mark[1], mark[2] - mark[0])
            generalized_marks.append([xval, yval, direction, mark[4]])
        else:
            generalized_marks.append([xval, yval])
    return np.array(generalized_marks)

def rotate_vector(vector, angle_degree):
    """Rotate a vector with given angle in degree."""
    angle_rad = math.pi * angle_degree / 180
    xval = vector[0]*math.cos(angle_rad) + vector[1]*math.sin(angle_rad)
    yval = -vector[0]*math.sin(angle_rad) + vector[1]*math.cos(angle_rad)
    return xval, yval

def rotate_centralized_marks(centralied_marks, angle_degree):
    """Rotate centralized marks with given angle in degree."""
    rotated_marks = centralied_marks.copy()
    for i in range(centralied_marks.shape[0]):
        mark = centralied_marks[i]
        rotated_marks[i, 0:2] = rotate_vector(mark[0:2], angle_degree)
        rotated_marks[i, 2:4] = rotate_vector(mark[2:4], angle_degree)
    return rotated_marks

def rotate_image(image, angle_degree):
    """Rotate image with given angle in degree."""
    rows, cols, _ = image.shape
    rotation_matrix = cv.getRotationMatrix2D((rows/2, cols/2), angle_degree, 1)
    return cv.warpAffine(image, rotation_matrix, (rows, cols))
