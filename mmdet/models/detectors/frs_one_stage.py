# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import DETECTORS
from .kd_one_stage import KnowledgeDistillationSingleStageDetector


@DETECTORS.register_module()
class FeatureRichnessScoreKDSingleStageDetector(KnowledgeDistillationSingleStageDetector):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.

    Args:
        teacher_config (str | dict): Config file path
            or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
        kd_loss_cfg (dict): Loss scale for balance multiple loss.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_config,
                 teacher_ckpt=None,
                 eval_teacher=True,
                 kd_loss_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, neck, bbox_head, teacher_config, teacher_ckpt, eval_teacher,
                         train_cfg, test_cfg, pretrained)
        if neck is not None:
            self.aux_net = nn.ModuleList([ConvModule(self.neck.out_channels, 
                                                     self.teacher_model.neck.out_channels,
                                                     3,
                                                     stride=1,
                                                     padding=1)for _ in range(self.neck.num_outs)])
        else:
            raise NotImplementedError('Network without neck is not supported.')
        # FRS loss setting
        self.kd_warm = kd_loss_cfg.kd_warm
        self.feat_loss_scale = kd_loss_cfg.feat_loss_scale
        self.cls_loss_scale = kd_loss_cfg.cls_loss_scale
        self.steps = 1

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        cls, reg = self.bbox_head(x)
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(img)
            teacher_cls, teacher_reg = self.teacher_model.bbox_head(teacher_x)
        losses = self.bbox_head.loss(cls, reg, gt_bboxes, gt_labels, img_metas,
                                     gt_bboxes_ignore=gt_bboxes_ignore)

        # compute kd loss
        kd_feat_loss, kd_cls_loss = 0, 0
        for layer in range(self.neck.num_outs):
            x_score = cls[layer].sigmoid()
            teacher_score = teacher_cls[layer].sigmoid()
            mask = torch.max(teacher_score, dim=1).values.detach()
            f_loss = torch.pow((teacher_x[layer] - self.aux_net[layer](x[layer])), 2)
            c_loss = F.binary_cross_entropy(x_score, teacher_score, reduction='none')

            kd_feat_loss += (f_loss * mask[:,None,:,:]).sum() / mask.sum()
            kd_cls_loss +=  (c_loss * mask[:,None,:,:]).sum() / mask.sum()

        kd_loss = kd_feat_loss * self.feat_loss_scale + kd_cls_loss * self.cls_loss_scale
        if self.steps < self.kd_warm:
            kd_loss *= self.steps / self.kd_warm
            self.steps += 1
        losses['kd_loss'] = kd_loss
        return losses
