import torch

from lbnl_fdd.models.informer import InformerClassifier
from lbnl_fdd.models.nonstationary_transformer import NonstationaryTransformerClassifier


x = torch.randn(8, 100, 20)

models = [
    InformerClassifier(n_features=20, seq_len=100, n_classes=5),
    NonstationaryTransformerClassifier(n_features=20, seq_len=100, n_classes=5),
]

for model in models:
    y = model(x)
    print(type(model).__name__, y.shape)