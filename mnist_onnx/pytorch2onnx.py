import torch
import torch.nn as nn
import torch.nn.functional as F



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
        return F.softmax(x, dim=1)
 
model = Net()   




model.load_state_dict(torch.load("../mnist_libtorch/python/mnist.pth",map_location=torch.device('cpu')))
batch_size = 1  #批处理大小
input_shape = (1,28,28)   #输入数据

# set the model to inference mode
model.eval()

x = torch.randn(batch_size,*input_shape)		# 生成张量
export_onnx_file = "mnist.onnx"					# 目的ONNX文件名


dynamic_axes = {'input': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'},  # variable lenght axes
                'output': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'}}

torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],		# 输入名
                    output_names=["output"],	# 输出名
                    dynamic_axes= dynamic_axes)
                    #dynamic_axes={"input":{0:"batch_size"},		# 批处理变量
                    #                "output":{0:"batch_size"}})
