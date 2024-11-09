import torch, os
import torchvision
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt

from model.mnist_model import VAE_MNIST

from .logger import Logger

def init_optimizer_mnist(model:VAE_MNIST, lr:float):
    optim = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )
    
    return optim

def plot_loss_curve(elbo_losses:list, kl_divergences:list, epoch:int, filename:int):
    
    # Plot the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(elbo_losses, label='ELBO Loss', color='blue')
    plt.plot(kl_divergences, label='KL Div', color='red')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"GAN Loss Curves up to Epoch {epoch+1}")
    plt.savefig(f'{filename}')
    plt.close()    

def vae_loss(reconstructed_image:torch.tensor, original_image:torch.tensor, 
             mu:torch.tensor, var:torch.tensor):
        
    criterion = torch.nn.BCELoss(reduction='sum')  
    
    reconstruction_loss = criterion(
        reconstructed_image, original_image
    )    
    
    kl_divergence = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
    
    return reconstruction_loss, kl_divergence

def test_vae(model:VAE_MNIST, 
             test_dataloader:torch.utils.data.DataLoader, 
             device:torch.device,
             cur_epoch:int, output_dir:str):
    
    model.eval()
    with torch.no_grad():
        original_images, _ = next(iter(test_dataloader))
        batch_size, _, h, w = original_images.shape  
        
        reconstructed_image, mu, var = model(original_images.view(batch_size, -1).to(device))      
        
        fig, ax = plt.subplots(2, 10, figsize=(15, 3))
        for i in range(10):
            ax[0, i].imshow(original_images[i].view(h, w).cpu(), cmap="gray")
            ax[1, i].imshow(reconstructed_image[i].view(h, w).cpu(), cmap="gray")
            ax[0, i].axis("off")
            ax[1, i].axis("off")            
            
        plt.suptitle("Top: Original Images | Bottom: Reconstructed Images")
        plt.savefig(f'{output_dir}/generated_image_epoch_{cur_epoch+1}.png')
        plt.close()           

def train_vae(model:VAE_MNIST, train_dataloader:torch.utils.data.DataLoader,
          test_dataloader:torch.utils.data.DataLoader,
          lr:float,
          num_epochs:int, output_dir:str              
              ):
    
    optim = init_optimizer_mnist(model, lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logger = Logger(output_dir)
    
    logger.log_message(f'Training Model -- {model._get_name()} on device == {device}')
    logger.log_line()
    
    model.to(device)    
    
    elbo_losses, kl_divergences = [], []
    
    for cur_epoch in range(num_epochs):   
        
        total_loss = 0.0
        total_kl_divergence = 0.0
        
        model.train()
             
        for batch_idx, (original_images, _) in enumerate(train_dataloader):     
            
            optim.zero_grad()
            
            batch_size = original_images.size(0)       
            reconstructed_image, mu, var = model(original_images.view(batch_size, -1).to(device))            
            reconstruction_loss, kl_div = vae_loss(reconstructed_image, 
                                                   original_images.view(batch_size, -1).to(device),
                                                   mu, var)
            
            loss = reconstruction_loss + kl_div
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            
            total_loss += loss.item()
            total_kl_divergence += kl_div.item()
            
        elbo_losses.append(total_loss / len(train_dataloader))
        kl_divergences.append(total_kl_divergence / len(train_dataloader))            
        
        logger.log_message(f'Epoch -- {cur_epoch}/{num_epochs} Loss -- {total_loss/len(train_dataloader):.4f} - KL Divergence -- {total_kl_divergence/len(train_dataloader)}')      
        
        if (cur_epoch + 1) % 10 == 0:

            plot_loss_curve(
                elbo_losses, kl_divergences, 
                cur_epoch, 
                f'{output_dir}/train_loss_curve_{cur_epoch+1}.png'
            )       
            
            test_vae(
                model, test_dataloader, device, cur_epoch, output_dir
            )
            
            torch.save(
                model.state_dict(), 
                f'{output_dir}/{model._get_name()}_ckpt_{cur_epoch+1}.pt'
            )
               