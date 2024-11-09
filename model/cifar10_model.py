import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator_CIFAR10(torch.nn.Module):
    def __init__(self, latent_dim:int, feature_maps:int, image_channels:int):
        super(Generator_CIFAR10, self).__init__()
        
        self.latent_dim = latent_dim
        self.generator_modules = torch.nn.Sequential(
            # (100, 1, 1) -- > (feat_maps * 8, 4, 4)
            torch.nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(feature_maps * 8),
            torch.nn.ReLU(),
            
            # (feat_maps * 8, 4, 4) -- > (feat_maps * 4, 8, 8)
            torch.nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(feature_maps * 4),
            torch.nn.ReLU(),
            
            # (feat_maps * 4, 4, 4) -- > (feat_maps * 2, 16, 16)
            torch.nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(feature_maps * 2),
            torch.nn.ReLU(),
            
            # (feat_maps * 2, 16, 16) -- > (3, 32, 32)
            torch.nn.ConvTranspose2d(feature_maps * 2, image_channels, 4, 2, 1, bias=False),
            torch.nn.Tanh(),                                    
        )
        
    def forward(self, latent_x:torch.tensor):
        return self.generator_modules(latent_x)
    
class Discriminator_CIFAR10(torch.nn.Module):
    def __init__(self, img_channels:int, feature_maps:int):
        super(Discriminator_CIFAR10, self).__init__()        
        
        self.discriminator_modules = torch.nn.Sequential(
            # (3, 32, 32) -- > (64, 16, 16)
            torch.nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(feature_maps),
            torch.nn.ReLU(),
            
            # (64, 16, 16) -- > (128, 8, 8)
            torch.nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(feature_maps * 2),
            torch.nn.ReLU(),
            
            # (128, 8, 8) -- > (256, 4, 4)
            torch.nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(feature_maps * 4),
            torch.nn.ReLU(),
            
            # (256, 4, 4) -- > (1, 1, 1)
            torch.nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x:torch.tensor):
        return self.discriminator_modules(x).view(-1, 1)        
    
class GAN_CIFAR10(torch.nn.Module):
    def __init__(self, latent_dim: int, 
                feature_maps: int, 
                image_channels: int):
        super(GAN_CIFAR10, self).__init__()
        
        self.generator = Generator_CIFAR10(latent_dim, feature_maps, 
                                               image_channels)
        
        self.discriminator = Discriminator_CIFAR10(image_channels, 
                                               feature_maps)
        
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        
    def forward_discriminator(self, inputs:torch.tensor):        
        return self.discriminator(inputs)
    
    def forward_generator(self, inputs:torch.tensor):        
        return self.generator(inputs)      