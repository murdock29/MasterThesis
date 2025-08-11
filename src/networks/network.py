from torch import nn
from copy import deepcopy


class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(self, model):
        super(LLL_Net, self).__init__()
        self.model = model

    def forward(self, x):
        """Applies the forward pass"""
        return self.model(x)

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False
