import torchvision.transforms as T
from torchvision.transforms import *
from copy import deepcopy


class Transform():
    def __init__(self, augs):
        # augs list of dict
        auglist = [*map(self.fromdict, augs)]

        self.transform = T.Compose(*[auglist])

    def __call__(self, x):
        x_other = deepcopy(x)
        x_other = self.transform(x)
        x = self.transform(x)
        return x, x_other

    def fromdict(self, dic):
        assert 'type' in dic, 'type must be specified'
        func = dic.pop('type')
        if 'randomapply' not in dic:
            return eval(func)(**dic)
        else:
            randomapply = dic.pop('randomapply')
            return T.RandomApply([eval(func)(**dic)], p=randomapply['p'])
