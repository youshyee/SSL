'''
sample cfg
'''
from yxy import path
################## ! essentials ###################
work_dir = f'{path.work}/mutual/simsiam'
load_from = None
resume_from = None
autoresume = False
total_epochs = 300
res = 224
gpu_batch = 64

################## ! model ###################
model_type = 'SimSiam'
model = dict(backbone='resnet50',
             projector=dict(in_dim=2048, hidden_dim=512, out_dim=2048),
             predictor=dict(in_dim=2048, hidden_dim=512, out_dim=2048))
################## !dataset ###################

aug = dict(
    type='Compose',
    transforms=[
        # dict(type='CenterCrop', size=res),
        dict(type='RandomResizedCrop', size=res, scale=(0.2, 1.0)),
        dict(type='RandomHorizontalFlip'),
        dict(type='ColorJitter',
             brightness=0.4,
             contrast=0.4,
             saturation=0.4,
             hue=0.1,
             randomapply=dict(p=0.8)),
        dict(type='RandomGrayscale', p=0.2),
        dict(type='GaussianBlur',
             kernel_size=res // 20 * 2 + 1,
             sigma=(0.1, 2.0),
             randomapply=dict(p=0.5)),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
    ])

workers_per_gpu = 16
dataset_type = 'ImageNet'
dataset = dict(root=f'{path.data}/imagenet', split='train')

################## !optimizer ###################
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict()
lr_config = dict(policy='CosineRestart',
                 by_epoch=True,
                 warmup='linear',
                 warmup_iters=1000,
                 warmup_ratio=1.0 / 10,
                 periods=[total_epochs // 10] * 10,
                 restart_weights=[1] * 10,
                 min_lr=0.001)

################## ! others ###################
log_level = 'INFO'
dist_params = dict(backend='nccl')
log_config = dict(interval=20,
                  hooks=[dict(type='TextLoggerHook'),
                         dict(type='TensorboardLoggerHook')])
checkpoint_config = dict(interval=5)
check4resume = dict(interval=6000)
workflow = [('train', 1)]
device_ids = range(8)
