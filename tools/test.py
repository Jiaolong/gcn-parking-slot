import os
import torch
import pprint
from psdet.utils import dist
from psdet.utils.config import get_config
from psdet.utils.common import get_logger, set_random_seed

from psdet.models.builder import build_model
from psdet.datasets.builder import build_dataloader

from eval_utils import eval_utils

def eval_single_model(model, test_loader, cfg, ckpt_file, logger, dist_test):
    logger.info('Evaluting {:s}'.format(str(ckpt_file)))
    # load checkpoint
    model.load_params_from_file(filename=ckpt_file, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation   
    eval_utils.eval_point_detection(cfg, model, test_loader, logger, dist_test=dist_test, 
                result_dir=cfg.output_dir, save_to_file=cfg.get('save_to_file', True)) 
    
def main():

    cfg = get_config()
    logger = get_logger(cfg.log_dir, cfg.tag)
    # log to file
    logger.info(pprint.pformat(cfg))
    
    if cfg.launcher == 'none':
        dist_test = False
    else:
        logger.info('Start distributed testing ...')
        cfg.batch_size, cfg.local_rank = dist.init_dist_pytorch(
            cfg.batch_size, cfg.local_rank, backend='nccl'
        )
        cfg.data.val.batch_size = cfg.batch_size
        dist_test = True


    if dist_test:
        total_gpus = dist.get_world_size()
        logger.info('total_batch_size: %d' % (total_gpus * cfg.batch_size))

    test_set, test_loader, sampler = build_dataloader(
            cfg.data.val, dist=dist_test, training=False, logger=logger)

    model = build_model(cfg.model)
    # logger.info(model)

    with torch.no_grad():
        if cfg.eval_all:
            ckpt_files = list(cfg.model_dir.glob('*.pth'))
            ckpt_files = sorted(ckpt_files, key=os.path.getmtime, reverse=True)
            for ckpt_file in ckpt_files:
                eval_single_model(model, test_loader, cfg, ckpt_file=ckpt_file, logger=logger, dist_test=dist_test)
        else:
            eval_single_model(model, test_loader, cfg, ckpt_file=cfg.ckpt, logger=logger, dist_test=dist_test)

if __name__ == '__main__':
    main()

