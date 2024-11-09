from dataset_utils.dataset_utils import load_data
from model.mnist_model import GAN_MNIST

from trainer.trainer_mnist import train_mnist

if __name__ == "__main__":
    
    mnist_model_kwargs = {
        "latent_variable_size":64,
        "discriminator_hidden_size":256, 
        "generator_hidden_size":256,
        "image_size":784, #28x28 --> flatten
    }
    
    dataset_kwargs = {
        "dataset_type":"mnist",
        "batch_size":16, 
        "shuffle":True
    }
    
    trainer_kwargs = {
        "num_epochs":200,
        "generator_lr":0.0002,
        "discriminator_lr":0.0002, 
        "output_dir":"mnist_GAN_trial"
        
    }
    
    train_dataloader, test_dataloader = load_data(**dataset_kwargs)    
    mnist_gan = GAN_MNIST(**mnist_model_kwargs)    
        
    train_mnist(
        mnist_gan, 
        train_dataloader, 
        test_dataloader,
        trainer_kwargs['generator_lr'],
        trainer_kwargs['discriminator_lr'],
        trainer_kwargs['num_epochs'],
        trainer_kwargs['output_dir']
    )
    
    # for data_items in train_dataloader:
    #     print(data_items[0].shape)
    #     exit(1)
    