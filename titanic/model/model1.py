import torch
from torch import nn

class Net(nn.Module):
    def __init__(self, inputs: int, outputs: int,dropout_prob: float) -> None:
        '''
            See the pdf with the diagram to understand the net nomenclature
        '''
        super().__init__()
        self.num_inputs = inputs
        self.outputs = outputs
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inputs, 20),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(10, outputs),
            nn.Sigmoid()
        )
        

    def forward(self, inputs: torch.Tensor, batch_num: int) -> torch.Tensor:
        if inputs.shape != (batch_num,self.num_inputs):
            raise ValueError(f'Bad Shape if inputs Input Shape:{inputs.shape} != {(batch_num,self.num_inputs)}')
        out = self.linear_relu_stack(inputs)
        return out

    def clean(self):
        self.state = self.state.detach()*0
        self.prev_output = self.prev_output.detach()*0
