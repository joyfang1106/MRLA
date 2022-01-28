_base_ = './faster_rcnn_r50la_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        # depth=101,
        layers=[3, 4, 23, 3], 
        init_cfg=dict(type='Pretrained',
                      checkpoint='./pretrained/r101_mrla_78.66.pth.tar')))
