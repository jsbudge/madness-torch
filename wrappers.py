import torch
import numpy as np


class SKLearnWrapper(object):

    def __init__(self, skobject):
        self.classifier = skobject

    def forward(self, x):
        try:
            return self.classifier.predict_proba(x)
        except:
            return self.classifier.predict(x)

    def fit(self, x, y):
        self.classifier.fit(x, y)


class TorchWrapper(object):

    def __init__(self, model):
        self.classifier = model

    def forward(self, x):
        return self.classifier(x)

