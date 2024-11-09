import torch

class Generator_MNIST(torch.nn.Module):
    def __init__(self, latent_size:int, hidden_size:int, image_size:int):
        super(Generator_MNIST, self).__init__()
        
        layers = []
        in_features = latent_size #784
        
        self.latent_size = latent_size
        
        self.generator_modules = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, image_size),
            torch.nn.Tanh()
        )
    
    def forward(self, latent_x:torch.tensor):
        return self.generator_modules(latent_x)

class Discriminat_MNIST(torch.nn.Module):
    def __init__(self, image_size:int, hidden_size:int):
        super(Discriminat_MNIST, self).__init__()
        
        layers = []
        in_features = image_size
        
        self.discriminator_modules = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x:torch.tensor):
        return self.discriminator_modules(x)

class GAN_MNIST(torch.nn.Module):
    def __init__(self, 
                 image_size:int, 
                 discriminator_hidden_size:int, 
                 latent_variable_size:int, 
                 generator_hidden_size:int,
                 ):
        super(GAN_MNIST, self).__init__()
        
        self.generator = Generator_MNIST(latent_variable_size, 
                                         generator_hidden_size, image_size)
        
        self.discriminator = Discriminat_MNIST(image_size, 
                                               discriminator_hidden_size)
        
    def forward_discriminator(self, inputs:torch.tensor):        
        return self.discriminator(inputs)
    
    def forward_generator(self, inputs:torch.tensor):        
        return self.generator(inputs)    


class VAE_MNIST(torch.nn.Module):
    def __init__(self, image_size:int, hidden_size:int, latent_dim:int):
        super(VAE_MNIST, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(image_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),                              
            torch.nn.Linear(hidden_size, latent_dim),
            torch.nn.ReLU()                                                                  
        )
        
        self.mu_layer = torch.nn.Linear(latent_dim, latent_dim)
        self.var_layer = torch.nn.Linear(latent_dim, latent_dim)
        
        self.decoder = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, hidden_size),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.LeakyReLU(0.2),                                         
                torch.nn.Linear(hidden_size, image_size),
                torch.nn.Sigmoid()
                # torch.nn.Tanh()
        )
        
    def reparametrize(self, mu:torch.tensor, var:torch.tensor):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def forward(self, original_image:torch.tensor):
        
        batch_size = original_image.shape[0]
        original_image = original_image.view(batch_size, -1)
        
        encoder_hidden_state = self.encoder(original_image)
        mu, var = self.mu_layer(encoder_hidden_state), self.var_layer(encoder_hidden_state)
        
        latent_x = self.reparametrize(mu, var)        
        generated_image = self.decoder(latent_x)

        return generated_image, mu, var
    
        
        