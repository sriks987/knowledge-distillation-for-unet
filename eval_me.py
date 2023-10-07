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
import matplotlib.pyplot as plt

if __name__ == "__main__":

    teacher = UNet(channel_depth = 16, n_channels = 3, n_classes=1)
    teacher.load_state_dict(torch.load("/content/CP_16_student3.pth"))
    teacher.eval().cuda()
    val_list = glob.glob('/content/val/*png')

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
        shuffle = False,
        transform = tf,
        ),
        batch_size = 1
    )    
    ll = []
    with torch.no_grad():
        for i,(img,gt) in enumerate(val_loader):
            print(img)
            if torch.cuda.is_available():
                img, gt = img.cuda(), gt.cuda()
            img, gt = Variable(img), Variable(gt)

            output = teacher(img)
            output = output.clamp(min = 0, max = 1)
            gt = gt.clamp(min = 0, max = 1)

            output_np = output.squeeze().cpu().detach().numpy()
            gt_np = gt.squeeze().cpu().detach().numpy()
            
            plt.imsave(f"output_{i}.png", output_np, cmap='gray')
            plt.imsave(f"gt_{i}.png", gt_np, cmap='gray')
            break

