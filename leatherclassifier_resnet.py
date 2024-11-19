import torch
import torch.nn as nn
from torch import optim
from torchvision import models
from torchvision.models import VGG16_BN_Weights, ResNet50_Weights, GoogLeNet_Weights, ResNet18_Weights

import numpy as np
import os

from dataset_v2 import DataSet_V2
from metrics import AccuracyScore
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


torch.set_printoptions(precision=2, sci_mode=False)

class LeatherClassifier_V5:
    def __init__(self, model, train_data_dir, test_data_dir):
        self.batch_size = 256
        self.num_workers = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.best_acc = 0
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.total_epoch = 100
        self.lr = 0.005
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_fn = AccuracyScore()
        self.opt = optim.SGD(
            params=[p for p in self.model.parameters() if p.requires_grad is True],
            lr=self.lr, weight_decay=0.0001
        )
        self.print_interval = 2
        self.model_dir = 'models'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        else:
            names = os.listdir(self.model_dir)
            if len(names) > 0:
                names.sort()
                name = names[-1]
                missing_keys, unexpected_keys = self.model.load_state_dict(
                    torch.load(os.path.join(self.model_dir, name)))
        self.model = self.model.to(self.device)  # 注意这一行要放在后面

    def save_model(self, epoch):
        # 模型保存
        if epoch == self.total_epoch:
            model_path = os.path.join(self.model_dir, "last.pth")
            torch.save(self.model.state_dict(), model_path)
        else:
            #model_path = os.path.join(self.model_dir, f"model_{epoch:04d}.pth")
            pass
        #torch.save(self.model.state_dict(), model_path)

    def save_best_model(self, acc):
        if self.best_acc <= acc:  # 等于的时候也更新
            self.best_acc = acc
            model_path = os.path.join(self.model_dir, "best.pth")
            torch.save(self.model.state_dict(), model_path)

    def train(self):
        # 1. 加载数据
        trainset = DataSet_V2(root_dir=self.train_data_dir,
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.num_workers,
                              istrainning=True)
        testset = DataSet_V2(root_dir=self.test_data_dir,
                             batch_size=self.batch_size,
                             shuffle=False,
                             num_workers=self.num_workers,
                             istrainning=False)

        for epoch in range(self.total_epoch):
            self.model.train(True)  # Sets the module in training mode.
            train_loss = []
            train_acc = []
            batch = 0
            for inputs, labels in trainset:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward
                output = self.model(inputs)
                loss = self.loss_fn(output, labels)

                # backward
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                acc = self.acc_fn(output, labels)

                train_loss.append(loss.item())
                train_acc.append(acc)
                if batch % self.print_interval == 0:
                    print(f'{epoch + 1}/{self.total_epoch} {batch} train_loss={loss.item()} -- acc={acc.item():.4f}')
                batch += 1

            test_loss = []
            test_acc = []
            batch = 0
            for inputs, labels in testset:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward
                output = self.model(inputs)
                loss = self.loss_fn(output, labels)
                acc = self.acc_fn(output, labels)

                test_loss.append(loss.item())
                test_acc.append(acc.item())
                if batch % self.print_interval == 0:
                    print(f'{epoch + 1}/{self.total_epoch} {batch} test_loss={loss.item()} --acc={acc.item():.4f}')
                batch += 1
            self.save_model(epoch)
            self.save_best_model(acc.item())

        print(f'{epoch} train mean loss {np.mean(train_loss):.4f} test mean loss {np.mean(test_loss):.4f}'
              f' train mean acc {np.mean(train_acc):.4f} test mean acc {np.mean(test_acc):.4f}')
        self.save_model(self.total_epoch)


if __name__ == '__main__':
    #vgg = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
    # googlenet = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    #resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    #resnet18 = models.resnet18()

    #vgg.classifier[3] = nn.Linear(in_features=4096, out_features=1000, bias=True)
    #vgg.classifier[6] = nn.Linear(in_features=1000, out_features=20, bias=True)
    # print(vgg)
    # googlenet.fc = nn.Linear(in_features=1024, out_features=20, bias=True)
    #resnet.fc = nn.Linear(in_features=2048, out_features=13, bias=True)
    resnet.fc = nn.Linear(in_features=512, out_features=13, bias=True)

    print(resnet)

    # 更多修改模型的示例：
    # del vgg.classifier[5]  # 删除classifier模块的某一层
    # vgg.classifier.append(nn.Softmax(dim=1)) #在classifier模块后面添加一个新的层
    # vgg.classifier.add_module('softmax', nn.Softmax(dim=1))  # 在classifier模块后面添加一个模块
    # vgg.features[13] = nn.Sequential() #将某一层置为空层（什么都不做，但是保留位置）
    # vgg.features[0].weight.requires_grad = False  # 某一层的参数冻结
    # print(vgg)

    train_data_dir = './data/train'
    test_data_dir = './data/test'
    model = LeatherClassifier_V5(resnet, train_data_dir, test_data_dir)
    model.train()
