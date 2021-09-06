import torch
from torch import nn
from torch.nn import functional as F

model_path = "../mnist_libtorch/python/mnist.pth"
device = torch.device('cpu')

 
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
 
model = Net()   


model_name = 'mnist_lenet'
# An instance of your model.
model.load_state_dict(torch.load(model_path , map_location='cpu'), strict=True)

# Evaluation mode
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 1, 28, 28)


def export_cpu(model, example):
    model = model.to("cpu")
    example = example.to("cpu")

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    # Save traced model
    traced_script_module.save("{}_model_cpu.pt".format(model_name))


def export_gpu(model, example):
    model = model.to("cuda")
    example = example.to("cuda")

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    # Save traced model
    traced_script_module.save("{}_model_gpu.pt".format(model_name))


export_cpu(model, example)

if torch.cuda.is_available():
    export_gpu(model, example)

