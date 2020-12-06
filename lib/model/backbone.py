from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3])


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3])


if __name__ == "__main__":

    import torch
    import torch.nn as nn
    m = resnet50()
    backbone = nn.Sequential(*list(m.children())[:-1])  # strip the cls head
    i = torch.randn(5, 3, 224, 224)
    out = backbone(i)
    print(out.shape)
