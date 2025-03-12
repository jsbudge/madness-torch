import torch
import numpy as np


class SKLearnWrapper(object):

    def __init__(self, skobject):
        self.classifier = skobject

    def forward(self, x):
        try:
            return self.classifier.predict_proba(x)[:, 0]
        except:
            return self.classifier.predict(x)[:, 0]

    def fit(self, x, y):
        self.classifier.fit(x, y)

    def __call__(self, x):
        return self.forward(x)


class TorchWrapper(object):

    def __init__(self, model):
        self.classifier = model

    def forward(self, x):
        return self.classifier(x)

    def __call__(self, x):
        return self.forward(x)

    def fit(self, x, y):
        pass


class SeasonalSplit:
    def __init__(self, random_state=None):
        self.random_state = random_state

    def split(self, X, y=None):
        seasons = list(set(X.index.get_level_values(1)))
        for s in seasons:
            tidxes = np.arange(X.shape)[X.index.get_level_values(1) == s]
            sidxes = np.arange(X.shape)[X.index.get_level_values(1) != s]
            yield tidxes, sidxes

    def get_n_splits(self, X, y=None, groups=None):
        return len(list(set(X.index.get_level_values(1))))

