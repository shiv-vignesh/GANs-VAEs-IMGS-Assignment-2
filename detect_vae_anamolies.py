import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset_utils.dataset_utils import load_data
from model.mnist_model import VAE_MNIST

def load_ckpt(model:VAE_MNIST, 
            weights_path:str, 
            device:torch.device):
    
    model.load_state_dict(
        torch.load(weights_path, map_location=device)
    )
    
def add_noise(images:torch.tensor, noise:float=0.5):
    images = images + noise * torch.rand_like(images)    
    return torch.clamp_(images, min=0.0, max=1.0)

def plot_error_distribution(clean_errors:list, corrupted_errors:list, output_dir:str):
    clean_errors = np.array(clean_errors)
    corrupted_errors = np.array(corrupted_errors)
    
    threshold = np.percentile(clean_errors, 95)  # 95th percentile of clean errors as threshold
    
    plt.hist(clean_errors, bins=50, alpha=0.6, color='blue', label='Normal Images')
    plt.hist(corrupted_errors, bins=50, alpha=0.6, color='red', label='Anomalous Images')
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.2f})')
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Reconstruction Error Distribution")
    plt.savefig(
        f'{output_dir}/reconstruction_error_distribution.png'
    )
    
def compute_reconstruction_error(model:VAE_MNIST, 
                                test_dataloader:torch.utils.data.DataLoader,
                                device:torch.device,
                                output_dir:str
                                ):
    
    model.eval()
    loss_func = torch.nn.BCELoss(reduction='sum')
    
    clean_errors, corrupted_errors = [], []
    
    for idx, (original_images, _) in enumerate(test_dataloader):
        with torch.no_grad():
            batch_size = original_images.size(0)       
            reconstructed_image, _, _ = model(original_images.view(batch_size, -1).to(device))                      
            clean_image_error = loss_func(reconstructed_image, original_images.view(batch_size, -1).to(device))
        
            corrupted_images = add_noise(original_images)        
            reconstructed_image, _, _ = model(corrupted_images.view(batch_size, -1).to(device))                      
            corrupted_image_error = loss_func(reconstructed_image, corrupted_images.view(batch_size, -1).to(device))      
            
            clean_errors.append(clean_image_error.item())
            corrupted_errors.append(corrupted_image_error.item())
    
    plot_error_distribution(
        clean_errors, corrupted_errors, output_dir
    )


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
    
    
    weights_path = "mnist_VAE/VAE_MNIST_ckpt_50.pt"
    output_dir = "mnist_VAE"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    _, test_dataloader = load_data(**dataset_kwargs)    
    mnist_vae = VAE_MNIST(**mnist_model_kwargs)
    
    mnist_vae.to(device)        
    
    if os.path.exists(weights_path):
        load_ckpt(mnist_vae, weights_path, device)
    
    compute_reconstruction_error(
        mnist_vae, test_dataloader, device, output_dir
    )