import os

import numpy as np
import torchvision
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from torchvision.transforms import transforms

from  PSPNet import PSPNet
from  Dataset import VOCSegmentationDataset
from test import test_net
from  test import visualize_output
import test



train_dir = 'C:/Users/86159/PycharmProjects/FCN语义分割/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
train_val_dir='C:/Users/86159/PycharmProjects/FCN语义分割/data/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt'
train_dataset=VOCSegmentationDataset(train_dir)
train_val_dataset=VOCSegmentationDataset(train_val_dir)



def Accuracy(output, label):
    # 将输出转换为预测的类别
    pred = torch.argmax(output, dim=1)
    # 计算预测正确的像素数
    correct_pixels = torch.sum(pred == label)
    # 计算总的像素数
    total_pixels = label.numel()
    # 计算精度
    accuracy = correct_pixels.float() / total_pixels
    return accuracy


def train_net(net, device, epochs=40, batch_size=1, lr=1e-5):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_val_loader= DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True)
    # 定义损失函数和优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,weight_decay=1e-4)
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion=nn.CrossEntropyLoss()
    writer = SummaryWriter("loss")
    checkpoint_path = 'network1.pth'
    epochh=0


    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epochh = checkpoint['epoch']+1
        print("Loaded checkpoint from epoch", epochh)



    # 训练模型
    for epoch in range(epochh,epochs):
        checkpoint_path = 'network1.pth'
        total_train_loss = 0
        total_train_val_loss = 0
        accuracy=0
        for i, (inputs,labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = torch.log_softmax(outputs,dim=1)
            visualize_output(outputs)
            loss = criterion(outputs, labels)
            total_train_loss+=loss.item()
            loss.backward()
            optimizer.step()
            #print(optimizer)
            print(f'Epoch {epoch+1}, batch {i+1} loss: {loss.item()} ')
            # 保存模型
            if (i+1)%100==0:
                print(f'Epoch {epoch + 1}, batch {i+1}, average loss: {total_train_loss/100} ')
                writer.add_scalar("train_loss", total_train_loss/100, i+epoch* 732)
                total_train_loss = 0
        #optimizer = torch.optim.Adam(net.parameters(), lr=lr * 0.9)
        if (epoch+1)%10==0:
            lr=0.1*lr

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)

        for i, (inputs, labels) in enumerate(train_val_loader, 0):
          with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            outputs = torch.log_softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            total_train_val_loss += loss.item()
            accuracy+=Accuracy(outputs,labels)
            # 保存模型
            if (i + 1) % 80 == 0:
                print(f'Epoch {epoch + 1}, average loss: {total_train_val_loss / 80} ,average accuracy: {accuracy/80}')
                writer.add_scalar("train_val_loss", total_train_val_loss / 80, 1 + epoch )
                writer.add_scalar("train_val_accuracy", accuracy/80, 1 + epoch)
                total_train_val_loss = 0
    writer.close()





if __name__ == '__main__':
    # 指定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建模型并移动到设备

    net = PSPNet(21).to(device)



    start=time.time()

    train_net(net, device)

    end=time.time()
    print(f'训练时间为 {end-start}s')




    # 训练模型

    test_net(net,device)
