import math
import torch
import numpy as np
from enum import Enum

class PointShape(Enum):
    """The point shape types used to pair two marking points into slot."""
    none = 0
    l_down = 1
    t_down = 2
    t_middle = 3
    t_up = 4
    l_up = 5

def direction_diff(direction_a, direction_b):
    """Calculate the angle between two direction."""
    diff = abs(direction_a - direction_b)
    return diff if diff < math.pi else 2*math.pi - diff

def detemine_point_shape(point, vector):
    """Determine which category the point is in."""
    BRIDGE_ANGLE_DIFF = 0.09757113548987695 + 0.1384059287593468
    SEPARATOR_ANGLE_DIFF = 0.284967562063968 + 0.1384059287593468

    vec_direct = math.atan2(vector[1], vector[0])
    vec_direct_up = math.atan2(-vector[0], vector[1])
    vec_direct_down = math.atan2(vector[0], -vector[1])
    if point[1][3] < 0.5: # shape
        if direction_diff(vec_direct, point[1][2]) < BRIDGE_ANGLE_DIFF:
            return PointShape.t_middle
        if direction_diff(vec_direct_up, point[1][2]) < SEPARATOR_ANGLE_DIFF:
            return PointShape.t_up
        if direction_diff(vec_direct_down, point[1][2]) < SEPARATOR_ANGLE_DIFF:
            return PointShape.t_down
    else:
        if direction_diff(vec_direct, point[1][2]) < BRIDGE_ANGLE_DIFF:
            return PointShape.l_down
        if direction_diff(vec_direct_up, point[1][2]) < SEPARATOR_ANGLE_DIFF:
            return PointShape.l_up
    return PointShape.none

def non_maximum_suppression(pred_points):
    """Perform non-maxmum suppression on marking points."""
    suppressed = [False] * len(pred_points)
    for i in range(len(pred_points) - 1):
        for j in range(i + 1, len(pred_points)):
            i_x = pred_points[i][1][0]
            i_y = pred_points[i][1][1]
            j_x = pred_points[j][1][0]
            j_y = pred_points[j][1][1]
            # 0.0625 = 1 / 16
            if abs(j_x - i_x) < 0.0625 and abs(j_y - i_y) < 0.0625:
                idx = i if pred_points[i][0] < pred_points[j][0] else j
                suppressed[idx] = True
    if any(suppressed):
        unsupres_pred_points = []
        for i, supres in enumerate(suppressed):
            if not supres:
                unsupres_pred_points.append(pred_points[i])
        return unsupres_pred_points
    return pred_points

def get_predicted_directional_points(prediction, point_thresh, boundary_thresh):
    """Get marking points from one predicted feature map.
        
        return:
            predicted_points: [x, y, direction, shape]
    """
    assert isinstance(prediction, torch.Tensor)
    predicted_points = []
    prediction = prediction.detach().cpu().numpy()
    for i in range(prediction.shape[1]):
        for j in range(prediction.shape[2]):
            if prediction[0, i, j] >= point_thresh:
                xval = (j + prediction[2, i, j]) / prediction.shape[2]
                yval = (i + prediction[3, i, j]) / prediction.shape[1]
                if not (boundary_thresh <= xval <= 1 - boundary_thresh
                        and boundary_thresh <= yval <= 1 -  boundary_thresh):
                    continue
                cos_value = prediction[4, i, j]
                sin_value = prediction[5, i, j]
                direction = math.atan2(sin_value, cos_value)
                # x, y, direction, shape
                marking_point = np.array([xval, yval, direction, prediction[1, i, j]])
                predicted_points.append((prediction[0, i, j], marking_point))
    return non_maximum_suppression(predicted_points)

def get_predicted_points(prediction, point_thresh, boundary_thresh):
    """Get marking points from one predicted feature map.
        
        return:
            predicted_points: [x, y]
    """
    assert isinstance(prediction, torch.Tensor)
    predicted_points = []
    prediction = prediction.detach().cpu().numpy()
    for i in range(prediction.shape[1]):
        for j in range(prediction.shape[2]):
            if prediction[0, i, j] >= point_thresh:
                xval = (j + prediction[1, i, j]) / prediction.shape[2]
                yval = (i + prediction[2, i, j]) / prediction.shape[1]
                if not (boundary_thresh <= xval <= 1 - boundary_thresh
                        and boundary_thresh <= yval <= 1 -  boundary_thresh):
                    continue
                marking_point = np.array([xval, yval])
                predicted_points.append((prediction[0, i, j], marking_point))
    return non_maximum_suppression(predicted_points)

def pass_through_third_point(marking_points, i, j, thresh):
    """See whether the line between two points pass through a third point."""
    x_1 = marking_points[i][1][0]
    y_1 = marking_points[i][1][1]
    x_2 = marking_points[j][1][0]
    y_2 = marking_points[j][1][1]
    for point_idx, point in enumerate(marking_points):
        if point_idx == i or point_idx == j:
            continue
        x_0 = point[1][0]
        y_0 = point[1][1]
        vec1 = np.array([x_0 - x_1, y_0 - y_1])
        vec2 = np.array([x_2 - x_0, y_2 - y_0])
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        if np.dot(vec1, vec2) > thresh:
            return True
    return False


def pair_marking_points(point_a, point_b):
    """See whether two marking points form a slot."""
    vector_ab = np.array([point_b[1][0] - point_a[1][0], point_b[1][1] - point_a[1][1]])
    vector_ab = vector_ab / np.linalg.norm(vector_ab)
    point_shape_a = detemine_point_shape(point_a, vector_ab)
    point_shape_b = detemine_point_shape(point_b, -vector_ab)
    if point_shape_a.value == 0 or point_shape_b.value == 0:
        return 0
    if point_shape_a.value == 3 and point_shape_b.value == 3:
        return 0
    if point_shape_a.value > 3 and point_shape_b.value > 3:
        return 0
    if point_shape_a.value < 3 and point_shape_b.value < 3:
        return 0
    if point_shape_a.value != 3:
        if point_shape_a.value > 3:
            return 1
        if point_shape_a.value < 3:
            return -1
    if point_shape_a.value == 3:
        if point_shape_b.value < 3:
            return 1
        if point_shape_b.value > 3:
            return -1

def calc_point_squre_dist(point_a, point_b):
    """Calculate distance between two marking points."""
    distx = point_a[0] - point_b[0]
    disty = point_a[1] - point_b[1]
    return distx ** 2 + disty ** 2
