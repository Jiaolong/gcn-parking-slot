import math
import torch
from torch import nn
from .utils import define_halve_unit, define_detector_block, YetAnotherDarknet, vgg16, resnet18, resnet50, MobileNetV3
from .gcn import GCNEncoder, EdgePredictor


class PointDetector(nn.modules.Module):
    """Detector for point without direction."""

    def __init__(self, cfg):
        super(PointDetector, self).__init__()
        self.cfg = cfg
        input_channel_size = cfg.input_channels
        depth_factor = cfg.depth_factor
        output_channel_size = cfg.output_channels
        
        if cfg.backbone == 'Darknet':
            self.feature_extractor = YetAnotherDarknet(input_channel_size, depth_factor)
        elif cfg.backbone == 'VGG16':
            self.feature_extractor = vgg16()
        elif cfg.backbone == 'resnet18':
            self.feature_extractor = resnet18()
        elif cfg.backbone == 'resnet50':
            self.feature_extractor = resnet50()
        elif cfg.backbone == 'MobileNetV3':
            self.feature_extractor = MobileNetV3()
        else:
            raise ValueError('{} is not implemented!'.format(cfg.backbone))
        
        layers_points = []
        layers_points += define_detector_block(16 * depth_factor)
        layers_points += define_detector_block(16 * depth_factor)
        layers_points += [nn.Conv2d(32 * depth_factor, output_channel_size,
                                    kernel_size=1, stride=1, padding=0, bias=False)]
        self.point_predictor = nn.Sequential(*layers_points)

        layers_descriptor = []
        layers_descriptor += define_detector_block(16 * depth_factor)
        layers_descriptor += define_detector_block(16 * depth_factor)
        layers_descriptor += [nn.Conv2d(32 * depth_factor, cfg.descriptor_dim,
                                        kernel_size=1, stride=1, padding=0, bias=False)]
        self.descriptor_map = nn.Sequential(*layers_descriptor)

        if cfg.use_gnn:
            self.graph_encoder = GCNEncoder(cfg.graph_encoder)

        self.edge_predictor = EdgePredictor(cfg.edge_predictor)

        if cfg.get('slant_predictor', None):
            self.slant_predictor = EdgePredictor(cfg.slant_predictor)

        if cfg.get('vacant_predictor', None):
            self.vacant_predictor = EdgePredictor(cfg.vacant_predictor)

    def forward(self, data_dict):
        img = data_dict['image']

        features = self.feature_extractor(img)  # [b, 1024, 16, 16]

        points_pred = self.point_predictor(features)
        points_pred = torch.sigmoid(points_pred)
        data_dict['points_pred'] = points_pred

        descriptor_map = self.descriptor_map(features)

        if self.training:
            marks = data_dict['marks']
            pred_dict = self.predict_slots(descriptor_map, marks[:, :, :2])
            data_dict.update(pred_dict)
        else:
            data_dict['descriptor_map'] = descriptor_map

        return data_dict

    def predict_slots(self, descriptor_map, points):
        descriptors = self.sample_descriptors(descriptor_map, points)
        data_dict = {}
        data_dict['descriptors'] = descriptors
        data_dict['points'] = points

        if self.cfg.get('slant_predictor', None):
            pred_dict = self.slant_predictor(data_dict)
            data_dict['slant_pred'] = pred_dict['edges_pred']

        if self.cfg.get('vacant_predictor', None):
            pred_dict = self.vacant_predictor(data_dict)
            data_dict['vacant_pred'] = pred_dict['edges_pred']

        if self.cfg.use_gnn:
            data_dict = self.graph_encoder(data_dict)

        pred_dict = self.edge_predictor(data_dict)

        data_dict['edge_pred'] = pred_dict['edges_pred']
        return data_dict

    def sample_descriptors(self, descriptors, keypoints):
        """ Interpolate descriptors at keypoint locations """
        b, c, h, w = descriptors.shape
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
        descriptors = torch.nn.functional.grid_sample(
            descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
        descriptors = torch.nn.functional.normalize(
            descriptors.reshape(b, c, -1), p=2, dim=1)
        return descriptors

    def get_targets_points(self, data_dict):
        points_pred = data_dict['points_pred']
        marks_gt_batch = data_dict['marks']
        npoints = data_dict['npoints']

        b, c, h, w = points_pred.shape
        targets = torch.zeros(b, c, h, w).cuda()
        mask = torch.zeros_like(targets)
        mask[:, 0].fill_(1.)

        for batch_idx, marks_gt in enumerate(marks_gt_batch):
            n = npoints[batch_idx].long()
            for marking_point in marks_gt[:n]:
                x, y = marking_point[:2]
                col = math.floor(x * w)
                row = math.floor(y * h)
                # Confidence Regression
                targets[batch_idx, 0, row, col] = 1.
                # Offset Regression
                targets[batch_idx, 1, row, col] = x * w - col
                targets[batch_idx, 2, row, col] = y * h - row

                mask[batch_idx, 1:3, row, col].fill_(1.)
        return targets, mask


class DirectionalPointDetector(nn.modules.Module):
    """Detector for point with direction."""

    def __init__(self, cfg):
        super(DirectionalPointDetector, self).__init__()
        self.cfg = cfg
        input_channel_size = cfg.input_channels
        depth_factor = cfg.depth_factor
        output_channel_size = cfg.output_channels
        self.feature_extractor = YetAnotherDarknet(input_channel_size, depth_factor)

        layers = []
        layers += define_detector_block(16 * depth_factor)
        layers += define_detector_block(16 * depth_factor)
        layers += [nn.Conv2d(32 * depth_factor, output_channel_size,
                             kernel_size=1, stride=1, padding=0, bias=False)]
        self.predict = nn.Sequential(*layers)

    def forward(self, data_dict):
        img = data_dict['image']
        prediction = self.predict(self.feature_extractor(img))
        # 4 represents that there are 4 value: confidence, shape, offset_x,
        # offset_y, whose range is between [0, 1].
        point_pred, angle_pred = torch.split(prediction, 4, dim=1)
        point_pred = torch.sigmoid(point_pred)
        angle_pred = torch.tanh(angle_pred)
        points_pred = torch.cat((point_pred, angle_pred), dim=1)
        data_dict['points_pred'] = points_pred
        return data_dict

    def get_targets(self, data_dict):
        marks_gt_batch = data_dict['marks']
        npoints = data_dict['npoints']
        batch_size = marks_gt_batch.size()[0]
        targets = torch.zeros(batch_size, self.cfg.output_channels,
                              self.cfg.feature_map_size,
                              self.cfg.feature_map_size).cuda()

        mask = torch.zeros_like(targets)
        mask[:, 0].fill_(1.)

        for batch_idx, marks_gt in enumerate(marks_gt_batch):
            n = npoints[batch_idx].long()
            for marking_point in marks_gt[:n]:
                x, y = marking_point[:2]
                col = math.floor(x * self.cfg.feature_map_size)
                row = math.floor(y * self.cfg.feature_map_size)
                # Confidence Regression
                targets[batch_idx, 0, row, col] = 1.
                # Makring Point Shape Regression
                targets[batch_idx, 1, row, col] = marking_point[3]  # shape
                # Offset Regression
                targets[batch_idx, 2, row, col] = x * 16 - col
                targets[batch_idx, 3, row, col] = y * 16 - row
                # Direction Regression
                direction = marking_point[2]
                targets[batch_idx, 4, row, col] = math.cos(direction)
                targets[batch_idx, 5, row, col] = math.sin(direction)

                mask[batch_idx, 1:6, row, col].fill_(1.)
        return targets, mask
