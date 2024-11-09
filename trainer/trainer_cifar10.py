import torch, os
import torchvision
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt

from model.cifar10_model import GAN_CIFAR10

from .logger import Logger

def init_optimizer_mnist(model:GAN_CIFAR10, generator_lr:float, discrim_lr:float):
    g_optim = torch.optim.Adam(
        model.generator.parameters(),
        lr=generator_lr,
        betas=(0.5, 0.999)
    )
    
    d_optim = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=discrim_lr,
        betas=(0.5, 0.999)
    )
    
    return g_optim, d_optim

def plot_loss_curve(d_losses:list, g_losses:list, epoch:int, filename:int):
    
    # Plot the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss', color='blue')
    plt.plot(g_losses, label='Generator Loss', color='red')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"GAN Loss Curves up to Epoch {epoch+1}")
    plt.savefig(f'{filename}')
    plt.close()    
    
def train_discriminator(model:GAN_CIFAR10, 
                        original_images:torch.tensor, criterion, 
                        d_optim:torch.optim, 
                        device:torch.device):
    
    d_optim.zero_grad()

    batch_size = original_images.size(0)
    real_labels = torch.ones(batch_size, 1).to(device) * 0.9
    fake_labels = torch.zeros(batch_size, 1).to(device) + 0.1
    
    outputs = model.forward_discriminator(original_images.to(device))
    d_loss_real = criterion(outputs, real_labels) 
    
    latent_x = torch.randn(batch_size,
                            model.generator.latent_dim, 1, 1).to(device)    
    
    fake_images = model.forward_generator(latent_x)
    outputs = model.forward_discriminator(fake_images.detach()) 
    
    d_loss_fake = criterion(outputs, fake_labels)       
    
    d_loss = (d_loss_real + d_loss_fake)/2
    d_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), 5.0)
    d_optim.step()

    return d_loss.item()      

def train_generator(model:GAN_CIFAR10, 
                    batch_size:int, criterion, 
                    g_optim:torch.optim, 
                    device:torch.device):
    
    g_optim.zero_grad()

    # real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)  

    latent_x = torch.randn(batch_size,
                            model.generator.latent_dim, 1, 1).to(device)
    
    fake_images = model.forward_generator(latent_x)
    outputs = model.forward_discriminator(fake_images)
    
    g_loss = - criterion(outputs, fake_labels)
    g_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 5.0)
    g_optim.step()

    return g_loss.item()    

def train_cifar10(model:GAN_CIFAR10,
    train_dataloader:torch.utils.data.DataLoader,
    test_dataloader:torch.utils.data.DataLoader,
    generator_lr:float, discriminator_lr:float,
    num_epochs:int, output_dir:str):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)    
    
    loss_criterion = torch.nn.BCELoss()    
    
    g_optim, d_optim = init_optimizer_mnist(model, generator_lr, discriminator_lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logger = Logger(output_dir)    
    
    logger.log_message(f'Training Model -- {model._get_name()} on device == {device}')
    logger.log_line()
    
    # Lists to store losses for plotting
    d_losses = []
    g_losses = []    
    
    batch_size = train_dataloader.batch_size    
    
    for cur_epoch in range(num_epochs):        
        for batch_idx, (original_images, _) in enumerate(train_dataloader):            
            d_loss = train_discriminator(model, original_images, loss_criterion, d_optim, device)
            g_loss = train_generator(model, batch_size, loss_criterion, g_optim, device)    

            d_losses.append(d_loss)
            g_losses.append(g_loss)

        epoch_d_loss = sum(d_losses)/len(train_dataloader)
        epoch_g_loss = sum(g_losses)/len(train_dataloader)        
        
        logger.log_message(f'Epoch -- {cur_epoch}/{num_epochs} -- Discriminator Loss: {epoch_d_loss:.4f} -- Generator Loss: {epoch_g_loss:.4f}')       
        
        if (cur_epoch + 1) % 50 == 0:

            plot_loss_curve(
                d_losses, g_losses, 
                cur_epoch, 
                f'{output_dir}/train_loss_curve_{cur_epoch+1}.png'
            ) 
            
            z = torch.randn(batch_size, model.generator.latent_dim, 1, 1).to(device)
            generated_images = model.generator(z)
            grid = torchvision.utils.make_grid(generated_images[:batch_size], nrow=4, normalize=True)
            plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
            plt.axis('off')
            plt.title(f"Generated Images at Epoch {cur_epoch+1}")
            plt.savefig(f'{output_dir}/generated_image_epoch_{cur_epoch+1}.png')
            plt.close()  
            
            torch.save(
                model.state_dict(), f'{output_dir}/{model._get_name()}_ckpt_{cur_epoch+1}.pt'
            )                       