import os
import torch, torchvision
import matplotlib.pyplot as plt

from dataset_utils.dataset_utils import load_data
from model.mnist_model import GAN_MNIST
from model.cifar10_model import GAN_CIFAR10

def load_mnist_gan_model_weights(model:GAN_MNIST, 
                                 weights_path:str,
                                 device:torch.device):
    
    return model.load_state_dict(
        torch.load(weights_path, map_location=device)
    )    
    
def alpha_only_interpolation(z1:torch.tensor, z2:torch.tensor, alpha):
    return (1 - alpha) * z1 + alpha * z2 

def interpolate_latent_spaces_mnist(model:GAN_MNIST, device:torch.device, output_dir:str, num_steps:int=10):
    
    model.eval()
    
    latent_dim = model.generator.latent_size
    z1 = torch.randn(1, latent_dim).to(device)
    z2 = torch.randn(1, latent_dim).to(device)
    
    interpolated_vectors = []
    
    for alpha in torch.linspace(0, 1, num_steps):
        z_interpolated = alpha_only_interpolation(z1, z2, alpha)
        interpolated_vectors.append(z_interpolated)
        
    with torch.no_grad():
        generated_images = [model.forward_generator(z).view(1, 28, 28) for z in interpolated_vectors]

    plt.figure(figsize=(15, 5))
    for i, img in enumerate(generated_images):
        plt.subplot(1, num_steps, i + 1)
        plt.imshow(img.cpu().squeeze().numpy(), cmap="gray")
        plt.axis("off")
        plt.title(f"α={torch.linspace(0, 1, num_steps)[i]:.2f}")
    plt.suptitle("Latent Space Interpolation between Two Generated Images")
    plt.savefig(
        f'{output_dir}/latent_space_interpolation.png'
    )      
    plt.close()

def interpolate_latent_spaces_cifar10(model:GAN_CIFAR10, device:torch.device, output_dir:str, num_steps:int=10):
    
    model.eval()
    
    latent_dim = model.generator.latent_dim
    z1 = torch.randn(1, latent_dim, 1, 1).to(device)
    z2 = torch.randn(1, latent_dim, 1, 1).to(device)
    
    interpolated_vectors = []
    
    for alpha in torch.linspace(0, 1, num_steps):
        z_interpolated = alpha_only_interpolation(z1, z2, alpha)
        interpolated_vectors.append(z_interpolated)
        
    with torch.no_grad():
        generated_images = [model.forward_generator(z).squeeze(0) for z in interpolated_vectors]

    generated_images = torch.stack(generated_images, dim=0)
    batch_size = len(generated_images)

    grid = torchvision.utils.make_grid(generated_images[:batch_size], nrow=5, normalize=True)
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.title(f"num_steps: {num_steps}")
    plt.savefig(f'{output_dir}/latent_space_interpolation.png')
    plt.close()  
          
def mnist_latent_interpolation():
    
    mnist_model_kwargs = {
        "latent_variable_size":64,
        "discriminator_hidden_size":256, 
        "generator_hidden_size":256,
        "image_size":784, #28x28 --> flatten
    }    
    
    weights_path = "mnist_GAN/mnist_gan_ckpt_200.pt"
    output_dir = "mnist_GAN"
    
    model = GAN_MNIST(**mnist_model_kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(weights_path):
        load_mnist_gan_model_weights(model, weights_path, device)
        
    model.to(device)
    
    interpolate_latent_spaces_mnist(
        model, device, output_dir
    )

def cifar10_latent_interpolation():
    
    model_kwargs = {
        "latent_dim": 100,
        "feature_maps": 64,
        "image_channels": 3
    }    
    
    weights_path = "cifar10_GAN/GAN_CIFAR10_ckpt_100.pt"
    output_dir = "cifar10_GAN"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model = GAN_CIFAR10(**model_kwargs)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(weights_path):
        print('loading model')
        model.load_state_dict(torch.load(
            weights_path, map_location=device
        ))
        
    model.to(device)
    
    interpolate_latent_spaces_cifar10(
        model, device, output_dir
    )
    

if __name__ == "__main__":
    
    # cifar10_latent_interpolation()
    mnist_latent_interpolation()