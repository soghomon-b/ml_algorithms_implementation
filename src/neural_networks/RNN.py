import torch
import torch.nn as nn



class CLTModelRNN(nn.Module):
    def __init__(self):
        super(CLTModelRNN, self).__init__()

    def initialise(
        self, hidden_size, output_size, batch_size, num_layers, input_size=100
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        rnn_out, hidden = self.rnn(input, hidden)
        output = self.h2o(rnn_out[:, -1, :])

        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
    