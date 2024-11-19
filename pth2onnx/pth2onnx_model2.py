import torch
import torch.nn as nn
import torchvision.models as models

if __name__ == '__main__':

    # resnet = models.resnet50()
    # resnet.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    resnet = models.resnet18()
    resnet.fc = nn.Linear(in_features=512, out_features=13, bias=True)
    state_dict = torch.load("../models/best.pth", map_location="cpu") #由于模型训练是在GPU上进行的，在CPU环境下加载需要加map_location="cpu"参数
    resnet.load_state_dict(state_dict)
    resnet.eval()

    x = torch.randn(1, 3, 224, 224)

    torch.onnx.export(resnet, x, "./leather_model2.onnx", export_params=True)