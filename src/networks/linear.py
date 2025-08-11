import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def linear(pretrained=False, weights=None,**kwargs):
    if pretrained:
        model = linear(**kwargs)
        state_dict = torch.hub.load_state_dict_from_url(weights)
        model.load_state_dict(state_dict)
    model = linear(**kwargs)
    return model
