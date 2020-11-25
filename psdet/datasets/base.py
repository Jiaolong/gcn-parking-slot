import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset

from .registry import DATASETS

@DATASETS.register
class BaseDataset(Dataset):
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.class_names = cfg.class_names
        self.root_path = Path(cfg.root_path)

    def __len__(self):
        raise NotImplementedError

    def forward(self, index):
        raise NotImplementedError

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            ret[key] = np.stack(val, axis=0)

        ret['batch_size'] = batch_size
        return ret
