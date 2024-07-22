"""This file contains the architectures and functions that I used in the project (.ipynb notebook)"""
"""Its purpose is mainly for testing and validating each one seperately"""
# Import library and set up environment
# General library
import os as os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Specific library needed for the project
import nibabel as nib
import random

# Define loss function
# Loss function: My dice loss
def dice_loss(prediction, truth):
    """Calculate 1 - generalized_dice_score
    The truth need to be one-hot encoded"""
    assert truth.max()<=1, "The truth need to be one-hot encoded"
    assert truth.min()>=0, "The truth need to be one-hot encoded"
    total_dim = truth.dim() #Dimension of input (4 for 2D image and 5 for 3D)
    # Count the number of class and get the weights
    # Calculate the loss
    numerator = torch.sum(truth*prediction, dim = tuple(range(2,total_dim)))
    denominator = torch.sum((truth**2+prediction**2), dim = tuple(range(2,total_dim)))
    dice = torch.where(denominator!=0,2* (numerator) / (denominator),denominator)
    dice = dice.mean(dim=1)
    return 1-dice.mean()

# Loss function: My generalized dice loss
def generalized_dice_loss(prediction, truth):
    """Calculate 1 - generalized_dice_score
    The truth need to be one-hot encoded"""
    assert truth.max()<=1, "The truth need to be one-hot encoded"
    assert truth.min()>=0, "The truth need to be one-hot encoded"
    epsilon=1e-6
    total_dim = truth.dim() #Dimension of input (4 for 2D image and 5 for 3D)
    # Count the number of class and get the weights
    weights = truth.sum(dim=tuple(range(2,total_dim)))
#     weights = weights**2
    weights = torch.where(weights!=0, 1/(weights), weights)
    # Normalize the weight
    weights = weights/weights.sum(dim=1).unsqueeze(1)
    # Calculate the loss
    numerator = torch.mean(truth*prediction, dim = tuple(range(2,total_dim)))
    denominator = torch.mean((truth**2+prediction**2), dim = tuple(range(2,total_dim)))
#     print(weights)
    dice = 1 - (2* (numerator+epsilon) / (denominator+epsilon))
#     print(dice.detach().cpu())
    dice = (dice*weights).sum(dim=1)
    return dice.mean()

# My dice score: Calculate dice score instead of dice loss
def dice_score_one_minus(prediction, truth):
    """Calculate 1 - dice_score
    The truth need to be one-hot encoded"""
    assert truth.max()<=1, "The truth need to be one-hot encoded"
    assert truth.min()>=0, "The truth need to be one-hot encoded"
    total_dim = truth.dim() #Dimension of input (4 for 2D image and 5 for 3D)
    # Count the number of class and get the weights
    # Calculate the loss
    prediction_result = prediction >= 0.5
    numerator = torch.sum(truth*prediction_result, dim = tuple(range(2,total_dim)))
    denominator = torch.sum((truth**2+prediction_result**2), dim = tuple(range(2,total_dim)))
    dice = 2* (numerator) / (denominator)
    dice = dice.mean(dim=1)
    return 1-dice.mean()

# Add elastic deformation as augmentation
import SimpleITK as sitk

"""Credit: The code is adapted from my MSc course!"""
def create_elastic_deformation(image, num_controlpoints, sigma):
    """
    We need to parameterise our b-spline transform
    The transform will depend on such variables as image size and sigma
    Sigma modulates the strength of the transformation
    The number of control points controls the granularity of our transform
    """
    #Transform zero np.array to sitk image
    itkimg = sitk.GetImageFromArray(np.zeros(image.shape))
    #Get total number of mesh point
    trans_from_domain_mesh_size = [num_controlpoints] * itkimg.GetDimension()
    #Initiate a transformation
    bspline_transformation = sitk.BSplineTransformInitializer(itkimg, trans_from_domain_mesh_size)
    #Get the transform parameter and convert them to np array
    params = np.asarray(bspline_transformation.GetParameters(), dtype=float)
    #Add randomness to transformation and set them back to sitk space
    params = params + np.random.randn(params.shape[0]) * sigma
    bspline_transformation.SetParameters(tuple(params))
    return bspline_transformation

def apply_elastic_deformation(image, bspline_transform, interpolation=sitk.sitkBSpline):
    # We need to choose an interpolation method for our transformed image, let's just go with b-spline
    resampler = sitk.ResampleImageFilter()    
    resampler.SetInterpolator(interpolation)
    # Let's convert our image to an sitk image
    sitk_image = sitk.GetImageFromArray(image)
    # Specify the image to be transformed: This is the reference image
    resampler.SetReferenceImage(sitk_image)
    resampler.SetDefaultPixelValue(0)
#     # Initialise the transform
#     bspline_transform = create_elastic_deformation(image, num_controlpoints, sigma)
    # Set the transform in the initialiser
    resampler.SetTransform(bspline_transform)
    # Carry out the resampling according to the transform and the resampling method
    out_img_sitk = resampler.Execute(sitk_image)
    # Convert the image back into a python array
    out_img = sitk.GetArrayFromImage(out_img_sitk)
    return out_img.reshape(image.shape)

# Add elastic augmentation
"""This is my implementation"""
class elastic_deform:
    def __init__(self, num_controlpoints=128, sigma=1):
        """num_controlpoints: tell how detail the grids used for transformation
        sigma: intensity of transformation"""
        self.num_controlpoints=num_controlpoints
        self.sigma=sigma
    def generate_transform(self, image):
        self.bspline_transform=create_elastic_deformation(image, self.num_controlpoints, self.sigma)
    def apply_transform_on_image(self, image):
        return torch.from_numpy(apply_elastic_deformation(image, self.bspline_transform, interpolation=sitk.sitkBSpline))
    def apply_transform_on_label(self, image):
        return torch.from_numpy(apply_elastic_deformation(image, self.bspline_transform, interpolation=sitk.sitkNearestNeighbor))

# Add noise augmentation
class add_noise:
    def __init__(self, noise_level=1.0):
        self.noise_level=noise_level
    def __call__(self, image):
        return image + torch.rand_like(image)*self.noise_level

# Add contrast augmentation
class add_contrast:
    def __init__(self, central_value=0.0, contrast_random_level=0.0, brightness_random_level=0.0):
        self.central_value=central_value
        self.contrast_random_level=contrast_random_level
        self.brightness_random_level=brightness_random_level
    def __call__(self, image):
        contrast = 1 + (torch.rand(1).item()*2-1)*self.contrast_random_level
        brightness = (torch.rand(1).item()*2-1)*self.brightness_random_level
        return (image-self.central_value)*contrast + brightness + self.central_value

# Focal loss
class FocalLoss(nn.Module): 
    def __init__(self, gamma=2.):
        """https://towardsdatascience.com/focal-loss-a-better-alternative-for-cross-entropy-1d073d92d075
        My implemetation of focal lost
        It take in probability (after softmax) and labels (after one hot encoded)
        And return the loss"""
        super().__init__()
        self.gamma = gamma
        self.eps = 1e-6
        
    def forward(self, input_tensor, target_tensor):
        input_tensor = torch.clamp(input_tensor, self.eps, 1. - self.eps)
        result = -((1-input_tensor)**self.gamma)*torch.log(input_tensor)*target_tensor
        result = result.mean(dim=(2,3,4))
        return result.mean()

# My dice loss with some modification
def dice_loss_weighted_TP_TN(prediction, truth, alpha=0.5):
    """Calculate 1 - generalized_dice_score
    The truth need to be one-hot encoded"""
    assert truth.max()<=1, "The truth need to be one-hot encoded"
    assert truth.min()>=0, "The truth need to be one-hot encoded"
    epsilon=1e-6
    total_dim = truth.dim() #Dimension of input (4 for 2D image and 5 for 3D)
    # Calculate the loss
    numerator_TP = torch.mean(truth*prediction, dim = tuple(range(2,total_dim)))
    denominator_TP = torch.mean((truth**2+prediction**2), dim = tuple(range(2,total_dim)))
    numerator_TN = torch.mean((1-truth)*(1-prediction), dim = tuple(range(2,total_dim)))
    denominator_TN = torch.mean(((1-truth)**2+(1-prediction)**2), dim = tuple(range(2,total_dim)))
    dice = alpha*2*(numerator_TP+epsilon) / (denominator_TP+epsilon) + (1-alpha)*2*(numerator_TN+epsilon) / (denominator_TN+epsilon)
    dice = dice.mean(dim=1)
    return 1-dice.mean()

# Combine focal with dice loss
class FocalDiceLoss(nn.Module): 
    def __init__(self, weights=[1.0,1.0,1.0], gamma=2., alpha=0.5):
        """Calculate dice loss but combine with the focal
        loss for adaptive weighting of hard to predict labels
        weights: the weight for each class
        gamma: use in focal loss (1-p)**gamma to balance between classes
        alpha: weight for TP and TN dice coefficient
        """
        super().__init__()
        assert alpha>=0.0 and alpha<=1.0, "alpha need to be within [0,1]"
        self.gamma = gamma
        self.weights = nn.Parameter(torch.tensor(weights).view(-1,3), requires_grad=False)        
        self.alpha = alpha
        self.eps = 1e-6
        
    def forward(self, input_tensor, target_tensor):
        assert target_tensor.max()<=1, "The truth need to be one-hot encoded"
        assert target_tensor.min()>=0, "The truth need to be one-hot encoded"
        total_dim = target_tensor.dim() #Dimension of input (4 for 2D image and 5 for 3D)
        # Calculate the dice loss for TP and TN
        numerator_TP = torch.mean(input_tensor*target_tensor, dim = tuple(range(2,total_dim)))
        denominator_TP = torch.mean((input_tensor**2+target_tensor**2), dim = tuple(range(2,total_dim)))
        numerator_TN = torch.mean((1-input_tensor)*(1-target_tensor), dim = tuple(range(2,total_dim)))
        denominator_TN = torch.mean(((1-input_tensor)**2+(1-target_tensor)**2), dim = tuple(range(2,total_dim)))
        dice = self.alpha*2*(numerator_TP+self.eps) / (denominator_TP+self.eps) + (1-self.alpha)*2*(numerator_TN+self.eps) / (denominator_TN+self.eps)
        # Use the focal loss formula to weight the dice coefficient and multiply it with custom weights by user
        result = -((1-dice)**self.gamma)*torch.log(dice)
        result = result*self.weights
        return result.mean()

def dice_loss_modified(prediction, truth):
    """Calculate 1 - generalized_dice_score
    The truth need to be one-hot encoded"""
    assert truth.max()<=1, "The truth need to be one-hot encoded"
    assert truth.min()>=0, "The truth need to be one-hot encoded"
    total_dim = truth.dim() #Dimension of input (4 for 2D image and 5 for 3D)
    # Count the number of class and get the weights
    # Calculate the loss
    prediction_modified = torch.where(prediction>=0.9, torch.tensor(1.0), prediction)
    prediction_modified = torch.where(prediction_modified<=0.1, torch.tensor(0.0), prediction_modified)
    numerator = torch.sum(truth*prediction_modified, dim = tuple(range(2,total_dim)))
    denominator = torch.sum((truth**2+prediction_modified**2), dim = tuple(range(2,total_dim)))
    dice = torch.where(denominator!=0,2* (numerator) / (denominator),denominator)
    dice = dice.mean(dim=1)
    return 1-dice.mean()

# Define a block for double convolution
class base_double_3D_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=None):
        '''
        in_channels - integer - the number of feature channels the first
                              convolution will receive
        out_channels - integer - the number of feature channels the last
                               convolution will output
        '''
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3), padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout, inplace=False) if dropout else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3,3,3), padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout, inplace=False) if dropout else nn.Identity(),
        )
    def forward(self, x):
        return self.double_conv(x)

# Main network
class My3DUNet_1(nn.Module):
    def __init__(self, depth = 4, in_channels=1, complexity=64, num_class=3, dropout=None):
        super().__init__()
        # Declare the dict
        modules = nn.ModuleDict()
        # First double comv layer
        feature_size=complexity
        modules[f"double_layers_down_{0}"] = base_double_3D_conv_block(in_channels=in_channels,
                                                                       out_channels=feature_size,
                                                                       dropout=dropout,)       
        # Downsample and then double conv the network for a depth total of time
        for d in range(1, depth+1, 1):
            # Reset in_channels and doubling the feature_size
            in_channels = feature_size
            feature_size *= 2
            # Downsampling first with stride conv layer
            modules[f"downsample_{d}"] = nn.MaxPool3d(kernel_size=2, stride=2)
            modules[f"double_layers_down_{d}"] = base_double_3D_conv_block(in_channels=in_channels,
                                                                           out_channels=feature_size,
                                                                           dropout=dropout)
        # Upsample and then double conv the network for a depth total of time
        for d in range(depth-1, -1, -1):
            # Reset in_channels and doubling the feature_size
            in_channels = feature_size
            feature_size //= 2
            # Downsampling first with stride conv layer
            modules[f"upsample_{d}"] = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=in_channels,
                    out_channels=feature_size,
                    kernel_size=2,
                    stride=2,
                ),
#                 nn.BatchNorm3d(feature_size),
#                 nn.ReLU(),                
            )
            modules[f"double_layers_up_{d}"] = base_double_3D_conv_block(in_channels=feature_size+feature_size,
                                                                         out_channels=feature_size,
                                                                         dropout=dropout,)
        # Last layer to project to number of class
        in_channels=feature_size
        feature_size=num_class
        modules[f"last_layers_to_classification"] = nn.Conv3d(in_channels=in_channels,
                                                              out_channels=feature_size,
                                                              kernel_size=1,
                                                              stride=1,
                                                              padding=0)
        self.my_modules = modules
        # Store the depth and the number of class
        self.depth=depth
        self.num_class=num_class

#         Initiate the weights
        self.initialize_weights()
        
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                                   nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                                   nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Store layer for skip connection
        stored_layer = {}
        # Downward path
        for d in range(0, self.depth, 1):
            stored_layer[d] = self.my_modules[f"double_layers_down_{d}"](x)
            x = self.my_modules[f"downsample_{d+1}"](stored_layer[d])
        # Bottleneck layer
        x = self.my_modules[f"double_layers_down_{self.depth}"](x)
        # Upward path
        for d in range(self.depth-1, -1, -1):
            x = self.my_modules[f"upsample_{d}"](x)
            x = torch.concat([x,stored_layer[d]], dim=1)            
            x = self.my_modules[f"double_layers_up_{d}"](x)
        x = self.my_modules[f"last_layers_to_classification"](x)
        return F.softmax(x, dim=1)

if __name__ == '__main__':
    """
    Test function
    Run the script to test all the function
    """

    # Test each function!!!   
    """Test dice_loss"""
    x = torch.rand(7,5,64,64,64)
    y = torch.randint(0, 2, size=(7,5,64,64,64), dtype=torch.long)
    y2 = y.to(torch.float)
    assert (dice_loss(y2,(1-y))-1)**2<1e-3, "dice_loss error"
    assert (dice_loss(y2,y)-0)**2<1e-3, "dice_loss error"

    """Test generalized_dice_loss"""
    x = torch.rand(1,3,64,128,128)
    y = torch.randint(0, 2, size=(1,3,64,128,128), dtype=torch.long)
    y2 = y.to(torch.float)
    assert (generalized_dice_loss(y2,(1-y))-1)**2<1e-3, "generalized_dice_loss error"
    assert (generalized_dice_loss(y2,y)-0)**2<1e-3, "generalized_dice_loss error"

    """Test dice_score_one_minus"""
    x = torch.rand(7,5,64,64,64)
    y = torch.randint(0, 2, size=(7,5,64,64,64), dtype=torch.long)
    y2 = y.to(torch.float)
    assert (dice_score_one_minus(y2,(1-y))-1)**2<1e-3, "dice_score_one_minus error"
    assert (dice_score_one_minus(y2,y)-0)**2<1e-3, "dice_score_one_minus error"

    """Test elastic_deform"""
    test_image_tensor = torch.rand(32, 64, 64)
    test = elastic_deform(num_controlpoints=128, sigma=10)
    test.generate_transform(test_image_tensor)
    transformed_image=test.apply_transform_on_image(test_image_tensor)
    transformed_label=test.apply_transform_on_label(test_image_tensor)
    assert transformed_image.size() == torch.Size([32, 64, 64]), "elastic_deform error"
    assert transformed_label.size() == torch.Size([32, 64, 64]), "elastic_deform error"

    """Test add_noise"""
    x = torch.rand(1, 1, 32, 64, 64)
    test = add_noise(1000.0)
    assert test(x).size() == torch.Size([1, 1, 32, 64, 64]), "add_noise error"

    """Test add_contrast"""
    x = torch.rand(1, 1, 32, 64, 64)
    test = add_contrast(0, 0.2, 400)
    assert test(x).size() == torch.Size([1, 1, 32, 64, 64]), "add_contrast error"

    """Test FocalLoss"""
    test = FocalLoss(2.0)
    x = torch.rand(1,3,64,128,128)
    y = torch.randint(0, 2, size=(1,3,64,128,128), dtype=torch.long)
    y2 = y.to(torch.float)
    assert (test(y2,y)-0)**2<1e-3, "FocalLoss error"

    """Test dice_loss_weighted_TP_TN"""
    x = torch.rand(7,5,64,64,64)
    y = torch.randint(0, 2, size=(7,5,64,64,64), dtype=torch.long)
    y2 = y.to(torch.float)
    assert (dice_loss_weighted_TP_TN(y2,(1-y))-1)**2<1e-3, "dice_loss_weighted_TP_TN error"
    assert (dice_loss_weighted_TP_TN(y2,y)-0)**2<1e-3, "dice_loss_weighted_TP_TN error"

    """Test FocalDiceLoss"""
    test = FocalDiceLoss(weights=[1,2,3], gamma=2.0, alpha=0.5)
    x = torch.rand(1,3,64,128,128)
    # print(x.max())
    y = torch.randint(0, 2, size=(1,3,64,128,128), dtype=torch.long)
    y2 = y.to(torch.float)
    assert (test(y2,y)-0)**2<1e-3, "FocalDiceLoss error"

    """Test dice_loss_modified"""
    x = torch.rand(1,3,64,128,128)
    # print(x.max())
    y = torch.randint(0, 2, size=(1,3,64,128,128), dtype=torch.long)
    y2 = y.to(torch.float)
    assert (dice_loss_modified(y2,(1-y))-1)**2<1e-3, "dice_loss_modified error"
    assert (dice_loss_modified(y2,y)-0)**2<1e-3, "dice_loss_modified error"

    """Test base_double_3D_conv_block"""
    test = base_double_3D_conv_block(3, 16, dropout=0.5)
    x = torch.rand((3, 3, 32, 64, 64))
    assert test(x).size() == torch.Size([3, 16, 32, 64, 64]), "base_double_3D_conv_block error"

    """Test My3DUNet_1"""
    depth = 4
    in_channels=1
    complexity=32
    num_class=3
    dropout=0.5
    test = My3DUNet_1(depth=depth, in_channels=in_channels, complexity=complexity, num_class=num_class, dropout=dropout)
    x = torch.rand((1, 1, 64, 128, 128)) # Test to find maximal batch_size that I should use
    result = test(x)
    assert result.size() == torch.Size([1, 3, 64, 128, 128]), "My3DUNet_1 error"

    # End the tests
    print("All tests passed")