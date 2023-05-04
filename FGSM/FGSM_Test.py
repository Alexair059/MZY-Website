import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.to(torch.float)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def check(img_path, std_path, model_path):
    std = np.array(Image.open(std_path)) / 255
    std = transforms.ToTensor()(std).unsqueeze(dim=0)
    MyNet = Net()
    MyNet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    MyNet.eval()

    raw_img = np.array(Image.open(img_path)) / 255
    img = transforms.ToTensor()(raw_img).unsqueeze(dim=0)

    threshold = 24
    d1 = (abs(img - std)).sum()

    if threshold < d1:
        return "侦测到明显修改"

    if torch.argmax(MyNet(img), dim=1) != torch.tensor([0]):
        return "神经网络检测为8"

    return r"攻击成功!Flag:SUSCTF{6radi3nt_1s_4ll_Y0u_Ne3d}"