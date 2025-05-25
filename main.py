import torch
from data.data_loader import load_dataset

def torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.random.seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def main():
    torch.manual_seed(42)

    train_loader,valid_loader=load_dataset()






if __name__ == '__main__':
    main()