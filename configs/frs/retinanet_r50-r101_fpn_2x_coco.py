_base_ = '../retinanet/retinanet_r50_fpn_2x_coco.py'

teacher_config = 'configs/retinanet/retinanet_r101_fpn_2x_coco.py'
teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth'

model = dict(
    type='FeatureRichnessScoreKDSingleStageDetector',
    teacher_config=teacher_config, 
    teacher_ckpt=teacher_ckpt,
    kd_loss_cfg=dict(
                    kd_warm=500, 
                    feat_loss_scale=0.005,
                    cls_loss_scale=0.02))

