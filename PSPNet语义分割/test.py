import numpy as np
import torchvision
from PIL.Image import Image
from matplotlib import pyplot as plt
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from  Dataset import VOCSegmentationDataset

test_dir='C:/Users/86159/PycharmProjects/FCN语义分割/data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
test_dataset=VOCSegmentationDataset(test_dir)
writer=SummaryWriter('logs')


colors = [
    (0, 0, 0),          # 背景
    (128, 0, 0),        # aeroplane
    (0, 128, 0),        # bicycle
    (128, 128, 0),      # bird
    (0, 0, 128),        # boat
    (128, 0, 128),      # bottle
    (0, 128, 128),      # bus
    (128, 128, 128),    # car
    (64, 0, 0),         # cat
    (192, 0, 0),        # chair
    (64, 128, 0),       # cow
    (192, 128, 0),      # diningtable
    (64, 0, 128),       # dog
    (192, 0, 128),      # horse
    (64, 128, 128),     # motorbike
    (192, 128, 128),    # person
    (0, 64, 0),         # potted plant
    (128, 64, 0),       # sheep
    (0, 192, 0),        # sofa
    (128, 192, 0),      # train
    (0, 64, 128),       # tv/monitor
]

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

def visualize_output(output):
    # 将输出和标签的张量转换为图像格式，并将像素值缩放到 0-1 的范围
    output = output.detach()
    #print(output.size())
    output = output.argmax(1).squeeze(0).cpu().numpy()
    color_image = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    for i in range(len(colors)):
        color_image[output == i] = colors[i]
    #print(color_image.shape)
    plt.imshow(color_image)
    plt.axis("off")
    plt.show()

def visualize_lable(lable):
    # 将输出和标签的张量转换为图像格式，并将像素值缩放到 0-1 的范围
    lable = lable.detach()
    #print(output.size())
    lable = lable.squeeze(0).cpu().numpy()
    #print(torch.sum(output))
    color_image = np.zeros((320, 480, 3), dtype=np.uint8)
    for i in range(len(colors)):
        color_image[lable == i] = colors[i]
    #print(color_image.shape)
    plt.imshow(color_image)
    plt.axis("off")
    plt.show()





def test_net(net, device, batch_size=1):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion=nn.CrossEntropyLoss()
    net.eval()
    accuracy=0
    val_loss=0

    with torch.no_grad():
        iou_list = []
        for i, (inputs, labels) in enumerate(test_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss+=loss
            accuracy+=Accuracy(outputs,labels)


            visualize_lable(labels)
            visualize_output(outputs)
            print(f'Test batch {i+1} loss: {loss.item()}')
            if (i + 1) % 724 == 0:
                print(f'Epoch  1, average loss: {val_loss / 724} ,average accuracy: {accuracy / 724}')

            # 可视化输出和标签
