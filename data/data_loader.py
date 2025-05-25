from torch.utils.data import DataLoader
from dataset import ImageDataset
import torchvision.transforms as torchvis


def training_preprocess(image_sz):
    return torchvis.Compose([
        torchvis.Resize((int(image_sz*1.2), int(image_sz*1.2))),
        torchvis.RandomCrop(image_sz),
        torchvis.RandomHorizontalFlip(p=0.5),
        torchvis.RandomRotation(degrees=10),
        torchvis.ColorJitter(brightness=0.2,contrast=0.2, saturation=0.2, hue=0.1),
        torchvis.GaussianBlur(kernel_size=3, sigma=(0.1,1.5)),
        
        torchvis.ToTensor(),
        torchvis.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])


def valid_preprocess(image_sz):
    return torchvis.Compose([
        torchvis.Resize((image_sz,image_sz)),
        
        torchvis.ToTensor(),
        torchvis.Normalize(mean = [0.485,0.456,0.406] ,std = [0.229,0.224,0.225])
    ])


def load_dataset():
    image_sz = 384
    batch_sz = 32
    
    train_set = ImageDataset(
        csv_file="csvs/train.csv",
        raw_dir="raw",
        transform=training_preprocess(image_sz),
        mode="train"
    )
    valid_set = ImageDataset(
        csv_file="csvs/val.csv",
        raw_dir="raw",
        transform=valid_preprocess(image_sz)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_sz,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_sz,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, valid_loader

