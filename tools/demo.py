import cv2
import time
import torch
import pprint
import numpy as np
from pathlib import Path

from psdet.utils.config import get_config
from psdet.utils.common import get_logger
from psdet.models.builder import build_model


def draw_parking_slot(image, pred_dicts):
    slots_pred = pred_dicts['slots_pred']

    width = 512
    height = 512
    VSLOT_MIN_DIST = 0.044771278151623496
    VSLOT_MAX_DIST = 0.1099427457599304
    HSLOT_MIN_DIST = 0.15057789144568634
    HSLOT_MAX_DIST = 0.44449496544202816

    SHORT_SEPARATOR_LENGTH = 0.199519231
    LONG_SEPARATOR_LENGTH = 0.46875
    junctions = []
    for j in range(len(slots_pred[0])):
        position = slots_pred[0][j][1]
        p0_x = width * position[0] - 0.5
        p0_y = height * position[1] - 0.5
        p1_x = width * position[2] - 0.5
        p1_y = height * position[3] - 0.5
        vec = np.array([p1_x - p0_x, p1_y - p0_y])
        vec = vec / np.linalg.norm(vec)
        distance =( position[0] - position[2] )**2 + ( position[1] - position[3] )**2 
        
        if VSLOT_MIN_DIST <= distance <= VSLOT_MAX_DIST:
            separating_length = LONG_SEPARATOR_LENGTH
        else:
            separating_length = SHORT_SEPARATOR_LENGTH
        
        p2_x = p0_x + height * separating_length * vec[1]
        p2_y = p0_y - width * separating_length * vec[0]
        p3_x = p1_x + height * separating_length * vec[1]
        p3_y = p1_y - width * separating_length * vec[0]
        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))
        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))
        p2_x = int(round(p2_x))
        p2_y = int(round(p2_y))
        p3_x = int(round(p3_x))
        p3_y = int(round(p3_y))
        cv2.line(image, (p0_x, p0_y), (p1_x, p1_y), (255, 0, 0), 2)
        cv2.line(image, (p0_x, p0_y), (p2_x, p2_y), (255, 0, 0), 2)
        cv2.line(image, (p1_x, p1_y), (p3_x, p3_y), (255, 0, 0), 2)

        #cv2.circle(image, (p0_x, p0_y), 3,  (0, 0, 255), 4)
        junctions.append((p0_x, p0_y))
        junctions.append((p1_x, p1_y))
    for junction in junctions:
        cv2.circle(image, junction, 3,  (0, 0, 255), 4)
    
    return image
    
def main():

    cfg = get_config()
    logger = get_logger(cfg.log_dir, cfg.tag)
    logger.info(pprint.pformat(cfg))

    model = build_model(cfg.model)
    logger.info(model)
    
    image_dir = Path(cfg.data_root) / 'testing' / 'outdoor-normal daylight'
    display = False

    # load checkpoint
    model.load_params_from_file(filename=cfg.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    
    if display:
        car = cv2.imread('images/car.png')
        car = cv2.resize(car, (512, 512))

    with torch.no_grad():

        for img_path in image_dir.glob('*.jpg'):
            img_name = img_path.stem
            
            data_dict = {} 
            image  = cv2.imread(str(img_path))
            image0 = cv2.resize(image, (512, 512))
            image = image0/255.

            data_dict['image'] = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).cuda()

            start_time = time.time()
            pred_dicts, ret_dict = model(data_dict)
            sec_per_example = (time.time() - start_time)
            print('Info speed: %.4f second per example.' % sec_per_example)

            if display:
                image = draw_parking_slot(image0, pred_dicts)
                image[145:365, 210:300] = 0
                image += car
                cv2.imshow('image',image.astype(np.uint8))
                cv2.waitKey(50)
                
                save_dir = Path(cfg.output_dir) / 'predictions'
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / ('%s.jpg' % img_name)
                cv2.imwrite(str(save_path), image)
    if display:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
