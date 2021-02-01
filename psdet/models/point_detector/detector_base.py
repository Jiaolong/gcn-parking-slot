import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..registry import POINT_DETECTOR
from .directional import PointDetector, DirectionalPointDetector

@POINT_DETECTOR.register
class PointDetectorBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.cfg = cfg
        
        if cfg.detector == 'DMPR':
            self.model = DirectionalPointDetector(cfg)
        else:
            self.model = PointDetector(cfg)
        
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
        return self.model.post_processing(data_dict)

    def get_training_loss(self, data_dict):
        return self.model.get_training_loss(data_dict)

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
