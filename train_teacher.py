import glob
import torch
import dataset
import numpy as np
from unet import UNet
import torch.nn as nn
from metrics import dice_loss
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR


def evaluate(teacher, val_loader):
    teacher.eval().cuda()

    criterion = nn.BCEWithLogitsLoss()
    ll = []
    with torch.no_grad():
        for i,(img,gt) in enumerate(val_loader):
            if torch.cuda.is_available():
                img, gt = img.cuda(), gt.cuda()
            img, gt = Variable(img), Variable(gt)
            output = teacher(img)
            output = output.clamp(min = 0, max = 1)
            gt = gt.clamp(min = 0, max = 1)
            loss = dice_loss(output, gt)
            ll.append(loss.item())

    
    mean_dice = np.mean(ll)
    print('Eval metrics:\n\tAverabe Dice loss:{}'.format(mean_dice))


def train_teacher(teacher, optimizer, train_loader):
    print(' --- teacher training')
    teacher.train().cuda()
    criterion = nn.BCEWithLogitsLoss()
    ll = []
    for i, (img, gt) in enumerate(train_loader):
        if torch.cuda.is_available():
            img, gt = img.cuda(), gt.cuda()
        
        img, gt = Variable(img), Variable(gt)

        output = teacher(img)
        output = output.clamp(min = 0, max = 1)
        gt = gt.clamp(min = 0, max = 1)
        loss = dice_loss(output, gt)
        ll.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    
    mean_dice = np.mean(ll)

    print("Average loss over this epoch:\n\tDice:{}".format(mean_dice))

def train_and_eval(teacher, channel_depth, optimizer, scheduler, train_loader, val_loader, num_epochs):
    for epoch in range(num_epochs):
        print(' --- teacher training: epoch {}'.format(epoch+1))
        train_teacher(teacher, optimizer, train_loader)

        # Evaluate for one epoch on validation set
        evaluate(teacher, val_loader)

        # Add checkpoint for epoch
        torch.save(teacher.state_dict(), f'/content/CP_teacher_{channel_depth}_{epoch+1}.pth')
        print("Checkpoint {} saved!".format(epoch+1))
        scheduler.step()

def create_teacher_stuff(teacher):
    '''
    Creates the teacher data structure and starts training
    '''  
    
    optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size = 100, gamma = 0.2)

    #load teacher and student model

    #NV: add val folder
    train_list = glob.glob('/content/train/*png')
    val_list = glob.glob('/content/val/*png')

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #2 tensors -> img_list and gt_list. for batch_size = 1 --> img: (1, 3, 320, 320); gt: (1, 1, 320, 320)
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
        shuffle = False,
        transform = tf,
        ),
        batch_size = 1
    )


    val_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
        shuffle = False,
        transform = tf,
        ),
        batch_size = 1
    )    

    return {
            "optimizer": optimizer, 
            "scheduler": scheduler,
            "train_loader": train_loader, 
            "val_loader": val_loader
            }


if __name__ == "__main__":
    channel_depth = 16
    num_epochs = 5
    teacher = teacher = UNet(channel_depth = channel_depth, n_channels = 3, n_classes=1)
    teacher_stuff = create_teacher_stuff(channel_depth)
    train_and_eval(teacher, channel_depth, **teacher_stuff, num_epochs= num_epochs)