import os
import sys
import torch
import pickle
import shutil
import logging
import datetime
import numpy as np
import subprocess
from pathlib import Path
import random as pyrandom
import torch.distributed as dist

from .dist import get_dist_info

def scan_upsample(points_array, input_rings=32, vertical_fov=26.8, bottom_angle=-24.8):
    im0, inds_e, inds_o = scan_to_range(points_array, input_rings, vertical_fov, bottom_angle)
    h, w, c = im0.shape
    points_new = []
    for i in range(h - 1):
        for j in range(w):
            d1, t1, v_angle1, h_angle1 = im0[i, j, :]
            d2, t2, v_angle2, h_angle2 = im0[i + 1, j, :]
            if d1 != 0 and d2 != 0:
                t = (t1 + t2) * 0.5
                d = (d1 + d2) * 0.5
                v_angle = (v_angle1 + v_angle2) * 0.5 
                h_angle = (h_angle1 + h_angle2) * 0.5
                x = np.sin(h_angle) * np.cos(v_angle) * d
                y = np.cos(h_angle) * np.cos(v_angle) * d
                z = np.sin(v_angle) * d
                point = np.array([x, y, z, t])
                points_new.append(point)
    points_new = np.array(points_new)
    points_hr = np.vstack((points_array, points_new))
    return points_hr

def scan_downsample(points_array, input_rings=64, vertical_fov=26.8, bottom_angle=-24.8, output_rings='even'):
    range_image, inds_e, inds_o = scan_to_range(points_array, input_rings, vertical_fov, bottom_angle)
    if output_rings == 'even':
        return points_array[inds_e,:4]
    elif output_rings == 'odd':
        return points_array[inds_o,:4]
    elif output_rings == 'even_or_odd':
        even = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        if even:
            return points_array[inds_e,:4]
        return points_array[inds_o,:4]
    elif output_rings == 'random':
        inds = inds_e + inds_o
        np.random.shuffle(inds)
        inds = inds[:int(len(inds) * 0.5)]
        return points_array[inds,:4]
    else:
        raise ValueError('Unknown output_rings value: %s', output_rings)
         
def range_to_scan(range_image, num_rings=64, vertical_fov=26.8, bottom_angle=-24.8):
    max_range = 80.0
    min_range = 2.0
    image_cols = 1024

    ang_res_x = 360.0 / float(image_cols) # horizontal resolution
    ang_res_y = vertical_fov / float(num_rings - 1) # vertical resolution
    row_ids = np.arange(num_rings)
    col_ids = np.arange(image_cols)
    v_angles = np.float32(row_ids * ang_res_y) + bottom_angle
    h_angles = np.float32(col_ids + 1 - image_cols / 2) * ang_res_x + 90
    v_angles = v_angles / 180.0 * np.pi
    h_angles = h_angles / 180.0 * np.pi
    
    range_image[:,:,0] *= max_range

    h, w, c = range_image.shape
    points = []
    inds_even = []
    inds_odd = []
    for i in range(h):
        for j in range(w):
            depth, intensity = range_image[i, j, :]
            if depth < min_range:
                continue
            h_angle = h_angles[j]
            v_angle = v_angles[i]
            x = np.sin(h_angle) * np.cos(v_angle) * depth
            y = np.cos(h_angle) * np.cos(v_angle) * depth
            z = np.sin(v_angle) * depth
            point = np.array([x, y, z, int(intensity)])
            points.append(point)
            idx = len(points) - 1
            if i % 2 == 0:
                inds_even.append(idx)
            else:
                inds_odd.append(idx)
    return np.array(points), inds_even, inds_odd

def scan_to_range(points_array, input_rings=64, vertical_fov=26.8, bottom_angle=-24.8, normalize=False):
    # range image size, depends on your sensor, i.e., VLP-16: 16x1800, OS1-64: 64x1024
    image_rows_full = input_rings
    max_range = 80.0
    min_range = 2.0
    image_cols = 1024

    ang_res_x = 360.0 / float(image_cols) # horizontal resolution
    ang_res_y = vertical_fov / float(image_rows_full - 1) # vertical resolution
    ang_start_y = bottom_angle

    # project points to range image
    # channels: range, intensity, horizon_angle, vertical_angle
    range_image = np.zeros((image_rows_full, image_cols, 4), dtype=np.float32)
    x = points_array[:,0]
    y = points_array[:,1]
    z = points_array[:,2]
    t = points_array[:,3]
    # find row id
    vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi
        
    relative_vertical_angle = vertical_angle - ang_start_y
    rowId = np.int_(np.round_(relative_vertical_angle / ang_res_y))

    # find column id
    horitontal_angle = np.arctan2(x, y) * 180.0 / np.pi
    colId = -np.int_((horitontal_angle - 90.0) / ang_res_x) + image_cols / 2;
    shift_ids = np.where(colId>=image_cols)
    colId[shift_ids] = colId[shift_ids] - image_cols
    # filter range
    thisRange = np.sqrt(x * x + y * y + z * z)
    thisRange[thisRange > max_range] = 0
    thisRange[thisRange < min_range] = 0

    if normalize:
        thisRange /= max_range

    # save range info to range image
    inds = []
    inds_odd_row = []
    inds_even_row = []
    for i in range(len(thisRange)):
        if rowId[i] < 0 or rowId[i] >= image_rows_full or colId[i] < 0 or colId[i] >= image_cols:
            continue
        range_image[int(rowId[i]), int(colId[i]), 0] = thisRange[i]
        range_image[int(rowId[i]), int(colId[i]), 1] = t[i]
        range_image[int(rowId[i]), int(colId[i]), 2] = vertical_angle[i] * np.pi / 180.0
        range_image[int(rowId[i]), int(colId[i]), 3] = horitontal_angle[i] * np.pi / 180.0

        if thisRange[i] > 0:
            inds.append(i)
            if rowId[i] % 2 == 0:
                inds_even_row.append(i)
            else:
                inds_odd_row.append(i)
    return range_image, inds_even_row, inds_odd_row

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers

def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

def set_random_seed(seed=3):
    pyrandom.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_logger(logdir, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = Path(logdir) / "run_{}.log".format(ts)
    file_hdlr = logging.FileHandler(str(file_path))
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger

def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()
    
    if rank != 0:
        return None
    
    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res)) 
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results
