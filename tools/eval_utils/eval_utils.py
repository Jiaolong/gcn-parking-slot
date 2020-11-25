import tqdm
import time
import pickle
import cv2
import numpy as np
import torch
from psdet.utils import common, dist
from psdet.models import load_data_to_gpu

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.model.post_processing.recall_thresh_list:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.model.post_processing.recall_thresh_list[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_point_detection(cfg, model, dataloader, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)
        
    dataset = dataloader.dataset
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.local_rank % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()
    
    if cfg.local_rank == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)

    start_time = time.time()
    point_pred_list = []
    point_gt_list = []
    slot_pred_list = []
    slot_gt_list = []
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict) 
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
       
            points_pred = pred_dicts['points_pred']
            point_pred_list += points_pred

            slots_pred = pred_dicts['slots_pred']
            slot_pred_list += slots_pred

            marks_gt_batch = batch_dict['marks']
            match_targets = batch_dict['match_targets']
            npoints = batch_dict['npoints']
            for b, marks_gt in enumerate(marks_gt_batch):
                n = npoints[b].long()
                marks = marks_gt[:n].cpu().numpy()
                point_gt_list.append(marks)
                
                match = match_targets[b][:n].cpu().numpy()
                slots = []
                for j, m in enumerate(match[:,0]):
                    if m >= 0 and m < n:
                        x1, y1 = marks[j,:2]
                        x2, y2 = marks[int(m),:2]
                        slot = np.array([x1, y1, x2, y2])
                        slots.append(slot)

                slot_gt_list.append(slots)

        if cfg.local_rank == 0:
            # progress_bar.set_postfix(disp_dict)
            progress_bar.update()
        
    if cfg.local_rank == 0:
        progress_bar.close()

    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Test finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.local_rank != 0:
        return {}
    
    dataset.evaluate_point_detection(point_pred_list, point_gt_list)
    logger.info('****************Point Detection Evaluation Done.*****************')
    dataset.evaluate_slot_detection(slot_pred_list, slot_gt_list)
    logger.info('****************Slot Detection Evaluation Done.*****************')
