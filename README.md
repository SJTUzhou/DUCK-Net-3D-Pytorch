# DUCK-Net-Pytorch
Unofficial implementation of DUCK-Net using Pytorch

Official Repo: [https://github.com/RazvanDu/DUCK-Net](https://github.com/RazvanDu/DUCK-Net)

Paper: [here](https://www.nature.com/articles/s41598-023-36940-5)



## DUCK-Net for 2D image segmentation tasks: duck_net.py

### official DuckNet model 

```
model = DuckNet(in_channels=3, 
                out_channels=1, 
                depth=5, 
                init_features=17, 
                normalization='batch', 
                interpolation='nearest', 
                out_activation='sigmoid', 
                use_multiplier=False)
model.apply(init_weights_with_kaiming_uniform) # default init is xaiver uniform
```
### Personal modified DuckNet model

```
# @ init_features=16: should be a power of 2 for better performance
# @ use_multiplier=True: for numerical stability
# @ normalization=None: reduce GPU memory usage
# @ out_activation=None: faster convergence when using Dice loss
model = DuckNet(in_channels=3, 
                out_channels=1, 
                depth=5, 
                init_features=16, 
                normalization=None, 
                interpolation='nearest', 
                out_activation=None, 
                use_multiplier=True)
```


## DUCK-Net for 3D medical image segmentation tasks: duck_net_3d.py (Performance not tested)

### Personal modified DuckNet3D model

```
# Personal modified DuckNet3D model: 
# @ init_features=16: should be a power of 2 for better performance
# @ depth=4: reduce depth for faster training and less GPU memory usage
# @ use_multiplier=True: for numerical stability
# @ normalization=None: reduce GPU memory usage
# @ out_activation=None: faster convergence when using Dice loss
model = DuckNet3D(in_channels=1, 
                out_channels=1, 
                depth=4, 
                init_features=16, 
                normalization=None, 
                interpolation='nearest', 
                out_activation=None, 
                use_multiplier=True)
```




