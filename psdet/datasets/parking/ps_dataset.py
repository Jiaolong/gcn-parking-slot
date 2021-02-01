import json
import math
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
class ParkingSlotDataset(BaseDataset):

    def __init__(self, cfg, logger=None):
        super(ParkingSlotDataset, self).__init__(cfg=cfg, logger=logger)
        
        assert(self.root_path.exists())
        
        if cfg.mode == 'train':
            data_dir = self.root_path / 'ps_json_label' / 'training'
        elif cfg.mode == 'val':
            data_dir = self.root_path / 'ps_json_label' / 'testing' / 'all'
 
        assert(data_dir.exists())
        
        self.json_files = [p for p in data_dir.glob('*.json')]
        self.json_files.sort()
        
        if cfg.mode == 'train': 
            # data augmentation
            self.image_transform = T.Compose([T.ColorJitter(brightness=0.1, 
                contrast=0.1, saturation=0.1, hue=0.1), T.ToTensor()])
        else:
            self.image_transform = T.Compose([T.ToTensor()])

        if self.logger:
            self.logger.info('Loading ParkingSlot {} dataset with {} samples'.format(cfg.mode, len(self.json_files)))
       
    def __len__(self):
        return len(self.json_files)
    
    def __getitem__(self, idx):
        json_file = Path(self.json_files[idx]) 
        # load json
        with open(str(json_file), 'r') as f:
            data = json.load(f)
        
        marks = np.array(data['marks'])
        if len(marks.shape) < 2:
            marks = np.expand_dims(marks, axis=0)

        max_points = self.cfg.max_points
        num_points = marks.shape[0]
        assert max_points >= num_points

        # centralize (image size = 600 x 600)
        marks[:,0:4] -= 300.5
        
        img_file = str(self.json_files[idx]).replace('.json', '.jpg').replace('ps_json_label', '')
        image = Image.open(img_file)
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

        marks = generalize_marks(marks, with_direction=self.cfg.with_direction)
        image = self.image_transform(image)
         
        # make sample with the max num points
        marks_full = np.full((max_points, marks.shape[1]), 0.0, dtype=np.float32)
        marks_full[:num_points] = marks
        match_targets = np.full((max_points, 2), -1, dtype=np.int32)
        
        slots = np.array(data['slots'])
        if slots.size != 0:
            if len(slots.shape) < 2:
                slots = np.expand_dims(slots, axis=0)
            for slot in slots:
                match_targets[slot[0] - 1, 0] = slot[1] - 1
                match_targets[slot[0] - 1, 1] = 0 # 90 degree slant

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
