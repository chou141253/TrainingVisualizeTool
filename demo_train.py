import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import os
import threading
import atexit
import time

from utils.web_render import WebRenderer

""" read training_info.txt file and return a dictionary """
def read_train_info(txt_path="training_info.txt"):
    lines = None
    with open(txt_path, "r") as f:
        lines = f.read().split("\n")
    training_info = {}
    for line in lines:
        if line == "":
            continue
        if line[0] == "#":
            continue
        line = line.replace(" ", "")
        datas = line.split("=")
        training_info[datas[0]] = datas[1]
    return training_info


def traing(epoch, model, dataloader, optimizer, criterion, training_info, canvas):
    model.train()
    for batch, (datas, labels) in enumerate(dataloader):
        if int(training_info["GPUs"])>0:
            datas, labels = datas.cuda(), labels.cuda()

        optimizer.zero_grad()
        outs = model(datas)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()

        if batch%update_every_batches == 0:
            # for record
            _, predicted = torch.max(outs.data, 1)
            correct = (predicted == labels).sum().item()
            accs["train"].append(100*correct/batch_size)
            losses["train"].append(loss.data.item())
            # for visualize
            if batch%(update_every_batches*5) == 0:
                show_val = True
            else:
                show_val = False
            canvas.updating(accs=accs["train"], 
                            losses=losses["train"], 
                            show_this=show_val, mode='train')


def test(epoch, model, dataloader, criterion, training_info, canvas):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for batch, (datas, labels) in enumerate(dataloader):
            if int(training_info["GPUs"])>0:
                datas, labels = datas.cuda(), labels.cuda()
            outs = model(datas)
            total_loss += criterion(outs, labels).data.item()
            _, predicted = torch.max(outs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100*correct/total
    test_avg_loss = total_loss/total
    # for record
    accs["test"].append(test_acc)
    losses["test"].append(test_avg_loss)
    canvas.updating(accs=accs["test"], 
                    show_this=True, mode='test')

def program_exit():
    """ 
    execute this function when program exit.
    """
    canvas.isbreak = True


if __name__ == "__main__":

    print("start to prepare model...")
    # Step 0. Reading training information from txt file.
    training_info = read_train_info()
    #print(training_info)


    batch_size = int(training_info['batch_size'])
    num_workers = int(training_info['num_workers'])
    update_every_batches = int(training_info['update_per_batches'])
    total_epoches = int(training_info['total_epoches'])

    # Training a cifar10 model,
    # reference : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    
    # Step 1.1 Prepare dataloader
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # Step 1.2 Create a model
    resnet18 = models.resnet18(pretrained=False)
    if int(training_info['GPUs'])>0:
        resnet18 = resnet18.cuda()


    # Step 1.3 Prepare Optimizer and Criterion for model.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)
    if int(training_info['GPUs'])>0:
        criterion, optimizer = criterion.cuda(), optimizer.cuda()

    print("prepare model done.\nstart to prepare web canvas...")
    # Step 1.4 Prepare web server
    canvas = WebRenderer(port=12345,
                         batch_size=batch_size,
                         sample_nums=len(trainloader.dataset), 
                         update_per_batches=update_every_batches, 
                         total_epoches=total_epoches, 
                         mode='auto', 
                         blank_size=70, 
                         epoch_pixel=30, 
                         max_vis_loss=7.6,
                         canvas_h=500,
                         x_ruler=5,
                         y_ruler=2)

    canvas_t = threading.Thread(target=canvas.start)
    canvas_t.deamon = True
    canvas_t.start()
    atexit.register(program_exit)

    time.sleep(3)

    print("start training...")
    # Step 2 training and testing
    accs = {"train":[], "test":[]}
    losses = {"train":[], "test":[]}
    for e in range(total_epoches):
        print("epoch[{}].  start training...".format(e))
        traing(epoch=e, 
               model=resnet18, 
               dataloader=trainloader, 
               optimizer=optimizer, 
               criterion=criterion, 
               training_info=training_info, 
               canvas=canvas)
        print("epoch[{}].  start testing...".format(e))
        test(epoch=e, 
             model=resnet18, 
             dataloader=testloader, 
             criterion=criterion, 
             training_info=training_info, 
             canvas=canvas)



