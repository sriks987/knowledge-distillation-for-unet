import torch
from torch.utils.data import Dataset
from data_augmentation import load_data, split_squares
import random
import numpy as np

class listDataset(Dataset):

    def __init__(self, ids, shuffle = True, transform = None, num_workers = 4):
        '''
        data loading
        '''
        if shuffle:
            random.shuffle(ids)

        self.nSamples = len(ids)
        self.lines = ids
        self.transform = transform
        #self.batch_size = batch_size -> Bs is mentioned in dataloader
        self.num_workers = num_workers
         

    def __getitem__(self, index):
        assert index <= len(self), 'Error: index out of bound'
        
        img_path = self.lines[index]
        print(img_path)
        img, gt = load_data(img_path)

        if self.transform is not None:
            img = self.transform(img)

        gt = np.array(gt)
        gt = gt = gt[:,:,0]
        gt = np.expand_dims(gt, axis = 2)
        gt = gt.transpose(2, 0, 1)

        gt = np.array(gt)
        i = img.float()
        g = torch.from_numpy(gt).float()
        return i, g
            

    def __len__(self):
        return self.nSamples