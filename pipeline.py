import eval_me
import dataset
import train_student
import train_teacher
import test_model

import glob
import torch
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

from unet import UNet

if __name__ == "__main__":

    teacher_channel_depth = 16
    student_channel_depth = 4
    num_epochs = 5

    # Creating teacher and student
    teacher = UNet(channel_depth = teacher_channel_depth, n_channels = 3, n_classes=1)
    teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    tacher_scheduler = StepLR(teacher_optimizer, step_size = 100, gamma = 0.2)

    student = UNet(channel_depth = student_channel_depth, n_channels = 3, n_classes=1)
    student_optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    student_scheduler = StepLR(student_optimizer, step_size = 100, gamma = 0.2)

    # Data loading
    train_list = glob.glob('/content/train/*png')
    val_list = glob.glob('/content/val/*png')
    test_list = glob.glob('/content/test/*png')

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

    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(test_list,
        shuffle = False,
        transform = tf,
        ),
        batch_size = 1
    )   

    # Check if trained teacher model already exists and load that
    try:
        teacher_state_dict_path = f'/content/CP_teacher_{teacher_channel_depth}_{num_epochs}'
        teacher.load_state_dict(torch.load(teacher_state_dict_path))
    except:
        # Train teacher
        train_teacher.train_and_eval(teacher = teacher,
                                 channel_depth=  teacher_channel_depth,
                                 optimizer = teacher_optimizer,
                                 train_loader = train_loader,
                                 val_loader = val_loader,
                                 num_epochs = num_epochs)
    

    # Evaluate teacher
    print("Evaluating teacher")
    eval_me.eval_model(teacher, val_loader)

    print("Performing knowledge distillation")
    # Train student
    train_student.train_and_eval_student(student = student,
                                         teacher = teacher,
                                         train_loader = train_loader,
                                         val_loader = val_loader,
                                         num_epochs = num_epochs)
    

    # Evaluate teacher on test set
    print("Evaluating teacher on test set")
    test_model.evaluate(teacher, test_loader)

    print("Evaluating student model on test set")
    test_model.evaluate(student, test_loader)

