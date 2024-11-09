import torch

from dataset_utils.dataset_utils import load_data
from model.cifar10_model import GAN_CIFAR10
from trainer.trainer_cifar10 import train_cifar10

if __name__ == "__main__":
    
    model_kwargs = {
        "latent_dim": 100,
        "feature_maps": 64,
        "image_channels": 3
    }
    
    dataset_kwargs = {
        "dataset_type":"cifar10",
        "batch_size":16, 
        "shuffle":True
    }  
    
    trainer_kwargs = {
        "num_epochs":200,
        "generator_lr":0.0002,
        "discriminator_lr":0.0002, 
        "output_dir":"cifar10_GAN"
        
    }    
    
    train_dataloader, test_dataloader = load_data(**dataset_kwargs)    
    
    model = GAN_CIFAR10(
        **model_kwargs
    )
    
    print(model)
    
    train_cifar10(
        model, train_dataloader, test_dataloader,
        trainer_kwargs['generator_lr'],
        trainer_kwargs['discriminator_lr'],
        trainer_kwargs['num_epochs'],
        trainer_kwargs['output_dir']        
    )
    
    # for data_items in train_dataloader:
    #     real_image = data_items[0]
    #     d_real_outputs = model.forward_discriminator(real_image)
        
    #     latent_x = torch.randn(dataset_kwargs['batch_size'],
    #                            model.generator.latent_dim, 1, 1)
    #     fake_image = model.forward_generator(latent_x)
        
    #     print(f'{real_image.shape} {fake_image.shape} {d_real_outputs.shape}')
    #     exit(1)