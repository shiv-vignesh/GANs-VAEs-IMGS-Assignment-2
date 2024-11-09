import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

def create_transforms_cifar10(apply_augmentation:bool=True):    
    if apply_augmentation:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32,padding=4),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            )
        ])
        
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5])
        ])

def create_transforms_mnist(normalize:bool=True):
    if normalize:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    else:
        return transforms.Compose([
            transforms.ToTensor()
        ])
        

def load_data(dataset_type:str="mnist", 
              batch_size:int=16, 
              shuffle:bool=True,
              normalize:bool=True):
    
    if dataset_type == "mnist":
        train_data = torchvision.datasets.MNIST(
            root='mnist_data',
            train=True, 
            download=True,
            transform=create_transforms_mnist(normalize)
        )
        
        test_data = torchvision.datasets.MNIST(
            root='mnist_data',
            train=False, 
            download=True,
            transform=create_transforms_mnist(normalize)
        )
        
    elif dataset_type == "cifar10":
        train_data = torchvision.datasets.CIFAR10(
            root='cifar10_data',
            download=True, 
            train=True,
            transform=create_transforms_cifar10(apply_augmentation=False)
        )
        
        test_data = torchvision.datasets.CIFAR10(
            root='cifar10_data',
            download=True, 
            train=True,
            transform=create_transforms_cifar10(apply_augmentation=False)
        )
    
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size,
        shuffle=shuffle
    )    
    
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size,
        shuffle=shuffle
    )   
    
    return train_dataloader, test_dataloader 
    