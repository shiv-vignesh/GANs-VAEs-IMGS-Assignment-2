import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from dataset_utils.dataset_utils import load_data
from model.mnist_model import VAE_MNIST
from detect_vae_anamolies import load_ckpt

def dimensionality_reductions(latents:np.array):
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    
    return latents_2d

def visualize_latent_space_2d(model:VAE_MNIST, 
                            test_dataloader:torch.utils.data.DataLoader,
                            device:torch.device,
                            output_dir:str):
    
    model.eval()
    
    labels, latents = [], []    
    for idx, (original_images, original_labels) in enumerate(test_dataloader):
        with torch.no_grad():
            batch_size = original_images.size(0)       
            
            _, mu, _ = model(original_images.view(batch_size, -1).to(device))                      
            latents.append(mu)
            labels.append(original_labels)    
            
    latents = torch.cat(latents).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    
    latents = dimensionality_reductions(latents)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, ticks=range(10), label="Digit Class")
    plt.title("2D Visualization of Latent Space (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig(
        f'{output_dir}/latents_space_2d.png'
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
        
    weights_path = "mnist_VAE_v2/VAE_MNIST_ckpt_50.pt"
    output_dir = "mnist_VAE_v2"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    _, test_dataloader = load_data(**dataset_kwargs)    
    mnist_vae = VAE_MNIST(**mnist_model_kwargs)
    
    mnist_vae.to(device)      
        
    if os.path.exists(weights_path):
        load_ckpt(mnist_vae, weights_path, device)
        
    visualize_latent_space_2d(
        mnist_vae, test_dataloader, device, output_dir
    )