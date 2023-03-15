import torch
from torch import nn

class Net(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        '''
            See the pdf with the diagram to understand the net nomenclature
        '''
        super().__init__()

    def forward(self, inputs):
        ...
    def clean(self):
        self.state = self.state.detach()*0
        self.prev_output = self.prev_output.detach()*0

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.state = self.state.to(*args, **kwargs) 
        self.prev_output = self.prev_output.to(*args, **kwargs) 
        return self