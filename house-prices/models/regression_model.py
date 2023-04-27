import torch
from torch import nn

class RModel(nn.Module):
    def __init__(self, inputs: int) -> None:
        '''
            See the pdf with the diagram to understand the net nomenclature
        '''
        super().__init__()
        self.num_inputs = inputs
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inputs, 50),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(50, 1),
        )
        

    def forward(self, inputs: torch.Tensor, batch_num: int) -> torch.Tensor:
        if inputs.shape != (batch_num,self.num_inputs):
            raise ValueError(f'Bad Shape if inputs Input Shape:{inputs.shape} != {(batch_num,self.num_inputs)}')
        out = self.linear_relu_stack(inputs)
        return out