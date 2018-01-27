
import io
import requests
from PIL import Image
from torch.autograd import Variable

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import models, transforms
from utils import progress_bar

import IPython


use_cuda = False


LABELS_URL = 'http://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
IMG_URL = 'http://s3.amazonaws.com/outcome-blog/wp-content/uploads/2017/02/25192225/cat.jpg'

net = models.squeezenet1_1(pretrained=True)

classifier = nn.Sequential(nn.Dropout(p=0.5), 
	nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1)), 
	nn.ReLU(),
	nn.AvgPool2d(13, stride=13, padding=0, ceil_mode=False, count_include_pad=True))

print (net)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


img_pil = Image.open("test/stopsign/5.jpg")

img_tensor = preprocess(img_pil)
img_tensor.unsqueeze_(0)
fc_out = net(Variable(img_tensor))

labels = {int(key):value for (key, value)
          in requests.get(LABELS_URL).json().items()}

print(labels[fc_out.data.numpy().argmax()])




trainset = torchvision.datasets.ImageFolder(root='train', transform=preprocess)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root='test',transform=preprocess)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(classifier.parameters(), lr=0.01)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = classifier(net.features(inputs))[:, :, 0, 0]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = classifier(net.features(inputs))[:, :, 0, 0]
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


for epoch in range(0, 200):
    train(epoch)
    test(epoch)