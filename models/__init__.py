"""
model list
"""
from .baseline import Baseline
from .model import MODEL

model_fn = {'baseline': Baseline, 'MODEL': MODEL}
