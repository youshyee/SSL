import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import *
from collections import OrderedDict


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError('{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def D(p, z, version='original'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out-
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d.
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.BatchNorm1d(out_dim))
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers.
        The dimension of h’s input and output (z and p) is d = 2048,
        and h’s hidden layer’s dimension is 512, making h a
        bottleneck structure (ablation in supplement).
        '''
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(inplace=True))
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing.
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimSiam(nn.Module):
    def __init__(self,
                 backbone='resnet50',
                 projector=dict(in_dim=512, hidden_dim=512, out_dim=2048),
                 predictor=dict(in_dim=2048, hidden_dim=512, out_dim=2048)):
        super(SimSiam, self).__init__()

        bone = eval(backbone)()
        boneout_feature = bone.fc.in_features
        self.backbone = nn.Sequential(*list(bone.children())[:-1])  # strip the cls head

        if projector is not None:
            projector['in_dim'] = boneout_feature
            self.projector = projection_MLP(**projector)

            # self.encoder = nn.Sequential(  # f encoder
            #     self.backbone, self.projector)
        # else:
        #     self.encoder = self.backbone

        self.predictor = prediction_MLP()

    def forward(self, x1, x2):
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        z1 = z1.squeeze(-1).squeeze(-1)
        z2 = z2.squeeze(-1).squeeze(-1)

        feature1 = z1.clone().detach()
        feature2 = z2.clone().detach()

        if hasattr(self, 'projector'):
            z1 = self.projector(z1)
            z2 = self.projector(z2)

        # z1, z2 = self.encoder(x1), self.encoder(x2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return L, feature1, feature2

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """

        data1 = data[0][0]
        data2 = data[0][1]  #(B,3,224,224)

        # print(data1.shape)
        # print(data2.shape)
        loss, feature1, feature2 = self(data1, data2)

        std1 = feature1.std(dim=0).mean()
        std2 = feature2.std(dim=0).mean()

        losses = dict(cc_loss=loss, std1=std1, std2=std2)
        loss, log_vars = parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data1))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """

        raise NotImplementedError
