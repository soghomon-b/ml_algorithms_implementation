import torch.nn as nn



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

    def initialise(self, input_size: tuple, output_size: int):
        in_ch, h, w = input_size
        self.cnn1 = nn.Conv2d(in_channels=in_ch, out_channels=32, kernel_size=3)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        self.flatten = nn.Flatten()
        self.output = nn.Linear(968256, output_size)
        print(input_size)

    def forward(self, input):
        input = self.relu(self.cnn1(input))
        input = self.pooling(input)
        input = self.relu(self.cnn2(input))
        input = self.pooling(input)
        flat = self.flatten(input)
        output = self.output(flat)

        return output