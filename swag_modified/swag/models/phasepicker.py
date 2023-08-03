import torch

class CNNNet(torch.nn.Module):

    def __init__(self, num_channels=1, min_lag = -0.5, max_lag = 0.5):
        super(CNNNet, self).__init__()
        from torch.nn import MaxPool1d, Conv1d, Linear
        self.relu = torch.nn.ReLU()
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.Hardtanh = torch.nn.Hardtanh(min_val = self.min_lag, max_val = self.max_lag)
        filter1 = 21
        filter2 = 15
        filter3 = 11

        linear_shape = 9600
        if num_channels == 1:
            linear_shape = 6400

        self.maxpool = MaxPool1d(kernel_size=2, stride=2)
        self.conv1 = Conv1d(num_channels, 32,
                            kernel_size=filter1, padding=filter1//2)
        self.bn1 = torch.nn.BatchNorm1d(32, eps=1e-05, momentum=0.1)
        # Output has dimension [200 x 32]

        
        self.conv2 = Conv1d(32, 64,
                            kernel_size=filter2, padding=filter2//2)
        self.bn2 = torch.nn.BatchNorm1d(64, eps=1e-05, momentum=0.1)
        # Output has dimension [100 x 64] 

        self.conv3 = Conv1d(64, 128,
                            kernel_size=filter3, padding=filter3//2)
        self.bn3 = torch.nn.BatchNorm1d(128, eps=1e-05, momentum=0.1)
        # Output has dimension [50 x 128]

        self.fcn1 = Linear(linear_shape, 512)
        self.bn4 = torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1)
  
        self.fcn2 = Linear(512, 512)
        self.bn5 = torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1)

        self.fcn3 = Linear(512, 1)

    def forward(self, x):
        # N.B. Consensus seems to be growing that BN goes after nonlinearity
        # That's why this is different than Zach's original paper.
        # First convolutional layer
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        # Second convolutional layer
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpool(x)
        # Third convolutional layer
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.maxpool(x)
        # Flatten
        x = x.flatten(1) #torch.nn.flatten(x)
        # First fully connected layer
        x = self.fcn1(x)
        x = self.relu(x)
        x = self.bn4(x)
        # Second fully connected layer
        x = self.fcn2(x)
        x = self.relu(x)
        x = self.bn5(x)
        # Last layer
        x = self.fcn3(x)
        # Force linear layer to be between +/- 0.5
        x = self.Hardtanh(x)
        return x

class PPicker:
    base = CNNNet
    args = list()
    kwargs = {"num_channels":1, "min_lag":-0.75, "max_lag":0.75}

class SPicker:
    base = CNNNet
    args = list()
    kwargs = {"num_channels":3, "min_lag":-0.85, "max_lag":0.85}