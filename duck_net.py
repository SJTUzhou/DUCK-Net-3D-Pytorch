import torch.nn as nn


class Normalization(nn.Module):
    def __init__(self, num_features, normalization='batch'):
        super(Normalization, self).__init__()
        self.num_features = num_features
        
        if normalization is None:
            self.norm = nn.Identity()
        elif normalization.lower() == 'batch':
            self.norm = nn.BatchNorm2d(num_features)
        elif normalization.lower() == 'instance':
            self.norm = nn.InstanceNorm2d(num_features)
        else:
            raise ValueError("Invalid normalization type. Supported types are None, 'batch', and 'instance'.")

    def forward(self, x):
        return self.norm(x)
    
    
    
class SeparatedConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, normalization='batch'):
        super(SeparatedConv2dBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1,kernel_size), stride=1, padding='same'),
            nn.ReLU(),
            Normalization(out_channels, normalization),
            nn.Conv2d(out_channels, out_channels, (kernel_size,1), stride=1, padding='same'),
            nn.ReLU(),
            Normalization(out_channels, normalization)
        )

    def forward(self, x):
        return self.block(x)
    
    
class MidScopeConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, normalization='batch'):
        super(MidScopeConv2dBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1),
            nn.ReLU(),
            Normalization(out_channels, normalization),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding='same', dilation=2),
            nn.ReLU(),
            Normalization(out_channels, normalization)
        )

    def forward(self, x):
        return self.block(x)
    
    
class WideScopeConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, normalization='batch'):
        super(WideScopeConv2dBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1),
            nn.ReLU(),
            Normalization(out_channels, normalization),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding='same', dilation=2),
            nn.ReLU(),
            Normalization(out_channels, normalization),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding='same', dilation=3),
            nn.ReLU(),
            Normalization(out_channels, normalization)
        )

    def forward(self, x):
        return self.block(x)
    
    
class ResnetConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, normalization='batch', multiplier=1.0):
        super(ResnetConv2dBlock, self).__init__()
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding='same', dilation=1),
            nn.ReLU()
        )
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1),
            nn.ReLU(),
            Normalization(out_channels, normalization),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1),
            nn.ReLU(),
            Normalization(out_channels, normalization)
        )
        self.norm = Normalization(out_channels, normalization)
        self.multiplier = multiplier

    def forward(self, x):
        x1 = self.block(x)
        x2 = self.shortcut(x)
        x = self.multiplier * (x1 + x2)
        return self.norm(x)
    
    
class DoubleConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, normalization='batch'):
        super(DoubleConv2dBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1),
            nn.ReLU(),
            Normalization(out_channels, normalization),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1),
            nn.ReLU(),
            Normalization(out_channels, normalization)
        )

    def forward(self, x):
        return self.block(x)
    
    
class DuckConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, normalization='batch', duck_multiplier=1.0, residual_block_multiplier=1.0):
        super(DuckConv2dBlock, self).__init__()
        self.norm1 = Normalization(in_channels, normalization)
        
        self.wide_scope = WideScopeConv2dBlock(in_channels, out_channels, kernel_size, normalization)
        self.mid_scope = MidScopeConv2dBlock(in_channels, out_channels, kernel_size, normalization)
        self.residual_block_1 = ResnetConv2dBlock(in_channels, out_channels, kernel_size, normalization, residual_block_multiplier)
        self.residual_block_2 = nn.Sequential(
            ResnetConv2dBlock(in_channels, out_channels, kernel_size, normalization, residual_block_multiplier),
            ResnetConv2dBlock(out_channels, out_channels, kernel_size, normalization, residual_block_multiplier)
        )
        self.residual_block_3 = nn.Sequential(
            ResnetConv2dBlock(in_channels, out_channels, kernel_size, normalization, residual_block_multiplier),
            ResnetConv2dBlock(out_channels, out_channels, kernel_size, normalization, residual_block_multiplier),
            ResnetConv2dBlock(out_channels, out_channels, kernel_size, normalization, residual_block_multiplier)
        )
        self.separated_block = SeparatedConv2dBlock(in_channels, out_channels, kernel_size, normalization)
        
        self.norm2 = Normalization(out_channels, normalization)
        self.duck_multiplier = duck_multiplier


    def forward(self, x):   
        x = self.norm1(x)
        
        x1 = self.wide_scope(x)
        x2 = self.mid_scope(x)
        x3 = self.residual_block_1(x)
        x4 = self.residual_block_2(x)
        x5 = self.residual_block_3(x)
        x6 = self.separated_block(x)
        
        x = self.duck_multiplier * (x1 + x2 + x3 + x4 + x5 + x6)
        return self.norm2(x)
        




class DuckNet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=5, init_features=32, normalization='batch', interpolation='nearest', out_activation='sigmoid', use_multiplier=False):
        """DUCK-Net for 2D image segmentation.
            reference paper: https://www.nature.com/articles/s41598-023-36940-5
        Args:
            in_channels (int): input channels of the image
            out_channels (int): output channels of the image
            depth (int): depth of the network. Defaults to 5.
            init_features (int): base number of features. Defaults to 32.
            normalization (str): normalization type ['batch','instance',None]. Defaults to 'batch'.
            interpolation (str): interpolation type, see nn.Upsample(). Defaults to 'nearest'.
            out_activation (str): output activation type ['sigmoid','softmax','relu',None]. Defaults to 'sigmoid'.
            use_multiplier (bool): use multiplier for numerical stability. Defaults to False.
        """
        super(DuckNet, self).__init__()
        
        duck_multiplier = 1.0 # multiplier for the duck block's addition operation
        residual_block_multiplier = 1.0 # multiplier for the residual block's addition operation
        self.addition_multiplier = 1.0 # multiplier in the encoder-decoder skip connection
        
        if use_multiplier: # use multiplier for numerical stability
            duck_multiplier = 1 / 6.0 
            residual_block_multiplier = 0.5
            self.addition_multiplier = 0.5
        
        self.depth = depth
        
        # Multi-scale blocks for input
        self.multi_scale_blocks = nn.ModuleList([nn.Conv2d(in_channels, init_features*2, kernel_size=2, stride=2, padding=0)])
        for d in range(1,depth):
            self.multi_scale_blocks.append(nn.Conv2d(init_features*(2**d), init_features*(2**(d+1)), kernel_size=2, stride=2, padding=0))
        
        # encoder downsampling blocks
        self.down_blocks = nn.ModuleList([nn.Conv2d(init_features, init_features*2, kernel_size=2, stride=2, padding=0)])
        for d in range(1,depth):
            self.down_blocks.append(nn.Conv2d(init_features*(2**d), init_features*(2**(d+1)), kernel_size=2, stride=2, padding=0))
        
        # encoder duck blocks
        self.encoder_duck_blocks = nn.ModuleList([DuckConv2dBlock(in_channels, init_features, normalization=normalization, duck_multiplier=duck_multiplier, residual_block_multiplier=residual_block_multiplier)])
        for d in range(1,depth):
            self.encoder_duck_blocks.append(DuckConv2dBlock(init_features*(2**d), init_features*(2**d), normalization=normalization, duck_multiplier=duck_multiplier, residual_block_multiplier=residual_block_multiplier))
            
        # bottleneck residual block
        self.bottleneck = nn.Sequential(
            ResnetConv2dBlock(init_features*(2**depth), init_features*(2**depth), kernel_size=3, normalization=normalization, multiplier=residual_block_multiplier),
            ResnetConv2dBlock(init_features*(2**depth), init_features*(2**depth), kernel_size=3, normalization=normalization, multiplier=residual_block_multiplier),
            ResnetConv2dBlock(init_features*(2**depth), init_features*(2**(depth-1)), kernel_size=3, normalization=normalization, multiplier=residual_block_multiplier),
            ResnetConv2dBlock(init_features*(2**(depth-1)), init_features*(2**(depth-1)), kernel_size=3, normalization=normalization, multiplier=residual_block_multiplier)
        )
            
        # decoder upsampling blocks
        self.up_blocks = nn.ModuleList(
            [nn.Upsample(scale_factor=2, mode=interpolation)] * depth
        )
        
        # decoder duck blocks
        self.decoder_duck_blocks = nn.ModuleList([DuckConv2dBlock(init_features, init_features, normalization=normalization, duck_multiplier=duck_multiplier, residual_block_multiplier=residual_block_multiplier)])
        for d in range(1,depth):
            self.decoder_duck_blocks.append(DuckConv2dBlock(init_features*(2**(d)), init_features*(2**(d-1)), normalization=normalization, duck_multiplier=duck_multiplier, residual_block_multiplier=residual_block_multiplier))

        self.out = nn.Conv2d(init_features, out_channels, kernel_size=1, stride=1, padding='same')
        
        if out_activation is None:
            self.out_activation = nn.Identity()
        elif out_activation.lower() == 'sigmoid':
            self.out_activation = nn.Sigmoid()
        elif out_activation.lower() == 'softmax':
            self.out_activation = nn.Softmax(dim=1)
        elif out_activation.lower() == 'relu':
            self.out_activation = nn.ReLU()
        else:
            raise ValueError("Invalid out_activation type. Supported types are 'sigmoid', 'softmax', 'relu', and None.")
        
        
    def forward(self, x):
        
        # downsample block for input
        p = [self.multi_scale_blocks[0](x),] # multi-resolution input
        for i in range(1,self.depth):
            xp = self.multi_scale_blocks[i](p[i-1])
            p.append(xp)
            
        # encoder
        t = [] # skip connections
        for d in range(self.depth):
            x = self.encoder_duck_blocks[d](x)
            t.append(x)
            x = self.down_blocks[d](x)
            x = self.addition_multiplier * (x + p[d])
            
        # bottleneck
        x = self.bottleneck(x)
        
        # decoder
        for d in range(self.depth-1, -1, -1): # [depth-1, depth-2, ..., 0]
            x = self.up_blocks[d](x)
            x = self.addition_multiplier * (x + t[d])
            x = self.decoder_duck_blocks[d](x)
            
        x = self.out(x)
        return self.out_activation(x)

            
        
if __name__ == "__main__":
    # Test the DuckNet model
    import torch
    
    def init_weights_with_kaiming_uniform(m):
        '''Initialize the weights of the model with kaiming uniform initialization.'''
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_uniform_(m.weight)
            
    def count_parameters(model):
        '''Count the number of trainable parameters in the model.'''
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
            
    x = torch.randn(2,3,256,256) # (batch_size, in_channels, height, width)
    
    # official DuckNet model 
    model = DuckNet(in_channels=3, out_channels=1, depth=5, init_features=17, normalization='batch', interpolation='nearest', out_activation='sigmoid', use_multiplier=False)
    model.apply(init_weights_with_kaiming_uniform) # default init is xaiver uniform
    
    
    # Personal modified DuckNet model: 
    # @ init_features=16: should be a power of 2 for better performance
    # @ use_multiplier=True: for numerical stability
    # @ normalization=None: reduce GPU memory usage
    # @ out_activation=None: faster convergence when using Dice loss
    # model = DuckNet(in_channels=3, out_channels=1, depth=5, init_features=16, normalization=None, interpolation='nearest', out_activation=None, use_multiplier=True)
    
    print("Number of parameters: ", count_parameters(model))
    
    y = model(x)
    print("Output shape: ", y.shape) # (batch_size, out_channels, height, width)
    print("Output min-max: ", y.min().item(), y.max().item())
    