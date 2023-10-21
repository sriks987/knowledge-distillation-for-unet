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
    print('Test metrics:\n\tAverabe Dice loss:{}'.format(mean_dice))


if __name__ == "__main__":

    teacher = UNet(channel_depth = 16, n_channels = 3, n_classes=1)
    teacher.load_state_dict(torch.load("/content/CP_16_4.pth"))
    test_list = glob.glob('/content/test/*png')

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test = torch.utils.data.DataLoader(
        dataset.listDataset(test_list,
        shuffle = False,
        transform = tf,
        ),
        batch_size = 1
    )    

    evaluate(teacher, test)

