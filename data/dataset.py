import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd



class ImageDataset(Dataset):
    def __init__(self,csv_file, raw_dir ,transform=None,mode='train'):
        self.dir = raw_dir
        self.transform = transform
        self.mode = mode

        self.datafr = pd.read_csv(csv_file)

        if self.mode == 'train':
            self.has_labels = True
        else:
            self.has_labels = False


    def __len__(self):
        return len(self.datafr)


    def __getitem__(self, row):
        img_id = self.datafr.iloc[row]['image_id']

        path = self.dir+'/'+img_id +'.png'

        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.has_labels:
            label = self.datafr.iloc[row]['label']
            return image, torch.tensor(label,dtype=torch.long)
        else:
            return image, img_id
