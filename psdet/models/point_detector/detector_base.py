import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..registry import POINT_DETECTOR
from .directional import PointDetector
from .post_process import get_predicted_points

@POINT_DETECTOR.register
class PointDetectorBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.cfg = cfg
        
        self.model = PointDetector(cfg)
        self.point_loss_func = nn.MSELoss().cuda()
        
    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                range_image_in
                range_image_gt
        Returns:
        """
        #print('data_dict:', data_dict)
        #t0 = time.time()
        data_dict = self.model(data_dict)
        #t1 = time.time()
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(data_dict)
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, ret_dicts = self.post_processing(data_dict)
            #t2 = time.time()
            #print('point detect:', t1 - t0)
            #print('slot detect:', t2 - t1)
            return pred_dicts, ret_dicts
     
    def post_processing(self, data_dict):
        ret_dicts = {}
        pred_dicts = {}
        
        points_pred = data_dict['points_pred']
        descriptor_map = data_dict['descriptor_map']
        
        points_pred_batch = []
        slots_pred = []
        for b, marks in enumerate(points_pred):
            points_pred = get_predicted_points(marks, self.cfg.point_thresh, self.cfg.boundary_thresh)
            points_pred_batch.append(points_pred)
         
            if len(points_pred) > 0:
                points_np = np.concatenate([p[1].reshape(1, -1) for p in points_pred], axis=0)
            else:
                points_np = np.zeros((self.cfg.max_points, 2))

            if points_np.shape[0] < self.cfg.max_points:
                points_full = np.zeros((self.cfg.max_points, 2))
                points_full[:len(points_pred)] = points_np
            else:
                points_full = points_np

            pred_dict = self.model.predict_slots(descriptor_map[b].unsqueeze(0), torch.Tensor(points_full).unsqueeze(0).cuda())
            edges = pred_dict['edges_pred'][0]
            n = points_np.shape[0]
            m = points_full.shape[0]
            
            slots = []
            for i in range(n):
                for j in range(n):
                    idx = i * m + j
                    score = edges[0, idx]
                    if score > 0.5:
                        x1, y1 = points_np[i,:2]
                        x2, y2 = points_np[j,:2]
                        slot = (score, np.array([x1, y1, x2, y2]))
                        slots.append(slot)

            slots_pred.append(slots)

        pred_dicts['points_pred'] = points_pred_batch
        pred_dicts['slots_pred'] = slots_pred
        return pred_dicts, ret_dicts

    def get_training_loss(self, data_dict):
        points_pred = data_dict['points_pred']
        targets, mask = self.model.get_targets_points(data_dict)

        disp_dict = {}
        
        loss_point = self.point_loss_func(points_pred * mask, targets * mask)
        
        edges_pred = data_dict['edges_pred']
        edges_target = torch.zeros_like(edges_pred)
        edges_mask = torch.zeros_like(edges_pred)

        match_targets = data_dict['match_targets']
        npoints = data_dict['npoints']

        for b in range(edges_pred.shape[0]):
            n = npoints[b].long()
            y = match_targets[b]
            m = y.shape[0]
            for i in range(n):
                t = y[i, 0]                
                for j in range(n):
                    idx = i * m + j
                    edges_mask[b, 0, idx] = 1
                    if j == t:
                        edges_target[b, 0, idx] = 1
                   
        loss_edge = F.binary_cross_entropy(edges_pred, edges_target, edges_mask)
        loss_all = self.cfg.losses.weight_point * loss_point + self.cfg.losses.weight_edge * loss_edge

        tb_dict = {
            'loss_all': loss_all.item(),
            'loss_point': loss_point.item(),
            'loss_edge': loss_edge.item()
        }
        return loss_all, tb_dict, disp_dict

    def load_params_from_file(self, filename, logger=None, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        
        if logger:
            logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if logger and 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict():
                if self.state_dict()[key].shape == model_state_disk[key].shape:
                    update_model_state[key] = val

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state and logger:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))
        
        if logger:
            logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))
        else:
            print('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)
        self.load_state_dict(checkpoint['model_state'])
        
        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
