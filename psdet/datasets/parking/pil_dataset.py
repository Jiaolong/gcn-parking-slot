import math
import cv2 as cv
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms as T

from psdet.datasets.base import BaseDataset
from psdet.datasets.registry import DATASETS
from psdet.utils.precision_recall import calc_average_precision, calc_precision_recall

from .process_data import boundary_check, overlap_check, rotate_centralized_marks, rotate_image, generalize_marks
from .utils import match_marking_points, match_slots 

@DATASETS.register
class PILDataset(BaseDataset):

    def __init__(self, cfg, logger=None):
        super(PILDataset, self).__init__(cfg=cfg, logger=logger)
        
        assert(self.root_path.exists())
        
        if cfg.mode == 'train':
            data_dir = self.root_path / 'train'
        elif cfg.mode == 'val':
            data_dir = self.root_path / 'test'
 
        label_dir = data_dir / 'label'
        assert(label_dir.exists())
        
        self.label_files = []
        for p in label_dir.glob('*.txt'):
            t, _, _ = self.get_label(str(p))
            if t == 1:
                self.label_files.append(p)

        self.label_files.sort()
        
        if cfg.mode == 'train': 
            # data augmentation
            self.image_transform = T.Compose([T.ColorJitter(brightness=0.1, 
                contrast=0.1, saturation=0.1, hue=0.1), T.ToTensor()])
        else:
            self.image_transform = T.Compose([T.ToTensor()])

        if self.logger:
            self.logger.info('Loading PIL {} dataset with {} samples'.format(cfg.mode, len(self.label_files)))
       
    def __len__(self):
        return len(self.label_files)
   
    def get_label(self, fname, split = '\t'):
        with open(fname) as f:
            lines = f.readlines()

            t = int(lines[0][0])
            angle = int(lines[1][:-1])
            box_list = []

            for line in lines[2:]:
                box_data = line.strip().split(split)
                box_list.append(box_data)
            f.close()

            return t, angle, box_list

    def __getitem__(self, idx):
        # load label
        t, angle, boxes = self.get_label(self.label_files[idx]) 
        
        slots = []
        marks = []
        for box in boxes:
            parked, x1, y1, x2, y2, x3, y3, x4, y4 = box
            p1 = [x1, y1]
            p2 = [x2, y2]
            if p1 not in marks:
                marks.append(p1)
            if p2 not in marks:
                marks.append(p2)
            idx_p1 = marks.index(p1)
            idx_p2 = marks.index(p2)
            slots.append([idx_p1, idx_p2])

        marks = np.array(marks).astype(np.float32)
        slots = np.array(slots)

        if len(marks.shape) < 2:
            marks = np.expand_dims(marks, axis=0)

        max_points = self.cfg.max_points
        num_points = marks.shape[0]
        assert max_points >= num_points

        img_file = str(self.label_files[idx]).replace('.txt', '.jpg').replace('label', 'image')
        image = Image.open(img_file)
        w, h = image.size
        
        # normalize marks
        marks[:,0] /= w
        marks[:,1] /= h

        image = image.resize((512,512), Image.BILINEAR)
        
        if self.cfg.mode == 'train+' and np.random.rand() > 0.2:
            angles = np.linspace(5, 360, 72)
            np.random.shuffle(angles)
            for angle in angles:
                rotated_marks = rotate_centralized_marks(marks, angle)
                if boundary_check(rotated_marks) and overlap_check(rotated_marks):
                    image = rotate_image(image, angle)
                    marks = rotated_marks
                    break

        image = self.image_transform(image)
         
        # make sample with the max num points
        marks_full = np.full((max_points, marks.shape[1]), 0.0, dtype=np.float32)
        marks_full[:num_points] = marks
        match_targets = np.full((max_points, 2), -1, dtype=np.int32)
        
        if slots.size != 0:
            if len(slots.shape) < 2:
                slots = np.expand_dims(slots, axis=0)
            for slot in slots:
                match_targets[slot[0], 0] = slot[1]
                match_targets[slot[0], 1] = 0 # 90 degree slant

        input_dict = {
                'marks': marks_full,
                'match_targets': match_targets,
                'npoints': num_points,
                'frame_id': idx,
                'image': image
                }
        
        return input_dict 

    def generate_prediction_dicts(self, batch_dict, pred_dicts):
        pred_list = []
        pred_slots = pred_dicts['pred_slots']
        for i, slots in enumerate(pred_slots):
            single_pred_dict = {}
            single_pred_dict['frame_id'] = batch_dict['frame_id'][i]
            single_pred_dict['slots'] = slots
            pred_list.append(single_pred_dict)
        return pred_list
     
    def evaluate_point_detection(self, predictions_list, ground_truths_list):
        precisions, recalls = calc_precision_recall(
            ground_truths_list, predictions_list, match_marking_points)
        average_precision = calc_average_precision(precisions, recalls)
        self.logger.info('precesions:')
        self.logger.info(precisions[-5:])
        self.logger.info('recalls:')
        self.logger.info(recalls[-5:])
        self.logger.info('Point detection: average_precision {}'.format(average_precision))

    def evaluate_slot_detection(self, predictions_list, ground_truths_list):
                
        precisions, recalls = calc_precision_recall(
            ground_truths_list, predictions_list, match_slots)
        average_precision = calc_average_precision(precisions, recalls)

        self.logger.info('precesions:')
        self.logger.info(precisions[-5:])
        self.logger.info('recalls:')
        self.logger.info(recalls[-5:])
        self.logger.info('Slot detection: average_precision {}'.format(average_precision))
