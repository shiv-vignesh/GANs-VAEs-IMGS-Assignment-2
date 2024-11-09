from dataset_utils.dataset_utils import load_data
from model.mnist_model import VAE_MNIST
from trainer.trainer_vae import train_vae

if __name__ == "__main__":
    
    mnist_model_kwargs = {
        "hidden_size":1024,
        "latent_dim":512,
        "image_size":784, #28x28 --> flatten
    }    
    
    dataset_kwargs = {
        "dataset_type":"mnist",
        "batch_size":16, 
        "shuffle":True,
        "normalize":False
    }    
    
    trainer_kwargs = {
        "num_epochs":50,
        "lr":1e-3,
        "output_dir":"mnist_VAE_v2"
        
    }    
    
    train_dataloader, test_dataloader = load_data(**dataset_kwargs)    
    
    mnist_vae = VAE_MNIST(**mnist_model_kwargs)
    
    train_vae(
        mnist_vae, train_dataloader, test_dataloader, trainer_kwargs["lr"], 
        trainer_kwargs['num_epochs'], trainer_kwargs['output_dir']
    )