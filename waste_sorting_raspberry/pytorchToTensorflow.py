import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable

import tensorflow as tf
import onnxruntime
import onnx
from onnx_tf.backend import prepare

#####################################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#####################################################################

# Load the trained model from file
trained_model = Net()
trained_model.load_state_dict(torch.load('wasteSorting.pth'))

# Export the trained model to ONNX
dummy_input = Variable(torch.randn(1, 1, 28, 28)) # one black and white 28 x 28 picture will be the input to the model
torch.onnx.export(learn.model.state_dict(), dummy_input, Path(os.getcwd())/'wasteSorting.onnx')

# Load the ONNX file
model = onnx.load(Path(os.getcwd())/'wasteSorting.onnx')

# Import the ONNX model to Tensorflow
tf_rep = prepare(model)

# Export the ONNX file to a Tensorflow compatible graph
tf_rep.export_graph(Path(os.getcwd())/'wasteSorting.pb')