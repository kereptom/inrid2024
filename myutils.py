import os
import torch
import cv2
import math
import numpy as np
from models.network_dm import RSTCANet as net
import torch.nn.functional as F

from models.residual_model_resdnet import *
from models.MMNet_TBPTT import *


import os
import torch


def initialize_deepdemosaick_model():
    model_path = 'models/deepdemosaick_models/bayer_noisy/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     
    model_params = torch.load(model_path+'model_best.pth')
    model = ResNet_Den(BasicBlock1, model_params[2], weightnorm=True)
    mmnet = MMNet(model, max_iter=model_params[1])
    for param in mmnet.parameters():
        param.requires_grad = False
    mmnet.load_state_dict(model_params[0])
    mmnet = mmnet.to(device)
    return mmnet

def deep_demosaic(patch, mmnet, device):
    with torch.no_grad():
        mmnet.eval()
        mosaic = torch.FloatTensor(patch).float()[None]
        # padding in order to ensure no boundary artifacts
        mosaic = F.pad(mosaic[:,None],(8,8,8,8),'reflect')[:,0]
        shape = mosaic[0].shape
        mask = generate_mask(shape, pattern='RGGB')
        M = torch.FloatTensor(mask)[None]
        mosaic = mosaic[...,None]*M

        mosaic = mosaic.permute(0,3,1,2)
        M = M.permute(0,3,1,2)

        p = Demosaic(mosaic.float(), M.float())
        if device.type == 'cuda':
            p.cuda_()
        xcur = mmnet.forward_all_iter(p, max_iter=mmnet.max_iter, init=True, noise_estimation=True)

        return (xcur[0].cpu().data.permute(1,2,0).numpy()[8:-8,8:-8] ) / 255.0


# Define the function to initialize the RSTCANet model
def initialize_rstcanet_model():
    # Args_rst class definition inside the function
    class Args_rst:
        def __init__(self):
            self.model_name = 'RSTCANet_L'
            self.task_current = 'models/rstcanet_models'
            self.n_channels = 3
            self.nc = 128
            self.window_size = 8
            self.num_heads = 8
            self.N = 8
            self.K = 4
            self.patch_size = 2

    # Create an instance of Args_rst
    args_rst = Args_rst()

    # Prepare parameters for the model
    num_heads = [args_rst.num_heads for _ in range(args_rst.K)]
    depths = [args_rst.N for _ in range(args_rst.K)]
    model_path = os.path.join(args_rst.task_current, args_rst.model_name + '.pth')
    
    # Determine the device (CUDA or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the RSTCANet model (assuming `net` is defined elsewhere in your code)
    model_rstcanet = net(
        in_nc=1, 
        out_nc=args_rst.n_channels, 
        patch_size=args_rst.patch_size, 
        nc=args_rst.nc, 
        window_size=args_rst.window_size, 
        num_heads=num_heads, 
        depths=depths
    )
    
    # Load pre-trained weights
    model_rstcanet.load_state_dict(torch.load(model_path), strict=True)
    
    # Set the model to evaluation mode
    model_rstcanet.eval()
    
    # Disable gradients for the model parameters
    for k, v in model_rstcanet.named_parameters():
        v.requires_grad = False
    
    # Move the model to the appropriate device
    model_rstcanet = model_rstcanet.to(device)

    # Return the initialized model
    return model_rstcanet


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def linear_scale(image):
    return (image - image.min()) / (image.max() - image.min())

def postprocess_image_simple(demosaiced_image):
    final_image = linear_scale(demosaiced_image)
    final_image = np.clip(final_image, 0, 1)
    final_image = (final_image * 255).astype(np.uint8)
    return final_image

# Add your demosaicing functions here
def nearest_neighbor_demosaic(raw_slice, bayer_pattern):
    h, w = raw_slice.shape
    rgb_image = np.zeros((h, w, 3), dtype=raw_slice.dtype)
    if bayer_pattern == 'RGGB':
        rgb_image[0::2, 1::2, 0] = raw_slice[0::2, 0::2]
        rgb_image[1::2, 0::2, 0] = raw_slice[0::2, 0::2]
        rgb_image[0::2, 0::2, 0] = raw_slice[0::2, 0::2]
        rgb_image[1::2, 1::2, 0] = raw_slice[0::2, 0::2]
        rgb_image[0::2, 1::2, 1] = raw_slice[0::2, 1::2]
        rgb_image[1::2, 0::2, 1] = raw_slice[1::2, 0::2]
        rgb_image[0::2, 0::2, 1] = (raw_slice[0::2, 1::2] + raw_slice[1::2, 0::2]) / 2
        rgb_image[1::2, 1::2, 1] = (raw_slice[0::2, 1::2] + raw_slice[1::2, 0::2]) / 2
        rgb_image[1::2, 1::2, 2] = raw_slice[1::2, 1::2]
        rgb_image[0::2, 1::2, 2] = raw_slice[1::2, 1::2]
        rgb_image[1::2, 0::2, 2] = raw_slice[1::2, 1::2]
        rgb_image[0::2, 0::2, 2] = raw_slice[1::2, 1::2]
    return rgb_image

def rgb2RGGB(im_rgb):
    G = im_rgb[:,:,1]
    R = im_rgb[:,:,0]
    B = im_rgb[:,:,2]

    im_bayer = np.empty((im_rgb.shape[0], im_rgb.shape[1]), np.float32)
    im_bayer[0::2, 0::2] = R[0::2, 0::2] # top left
    im_bayer[0::2, 1::2] = G[0::2, 1::2] # top right
    im_bayer[1::2, 0::2] = G[1::2, 0::2] # bottom left
    im_bayer[1::2, 1::2] = B[1::2, 1::2] # bottom right

    return im_bayer

# Function to create Bayer mask
def create_bayer_mask(H, W, device):
    mask = torch.zeros((1, H, W, 3), device=device)
    mask[:, 0::2, 0::2, 0] = 1  # R
    mask[:, 0::2, 1::2, 1] = 1  # G
    mask[:, 1::2, 0::2, 1] = 1  # G
    mask[:, 1::2, 1::2, 2] = 1  # B
    return mask.reshape(1, H*W, 3)

def create_complementary_bayer_mask(H, W, device):
    # Create the original Bayer mask
    original_mask = create_bayer_mask(H, W, device)
    # Invert the mask to get the complementary mask
    complementary_mask = 1 - original_mask
    return complementary_mask    
    

def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
        

def add_noise_power(image, snr_db):
    snr_linear = 10 ** (snr_db / 10)  # Convert SNR from dB to linear (power version)
    signal_power = np.mean(image ** 2)  # Calculate the power of the signal
    noise_power = signal_power / snr_linear  # Calculate the power of the noise
    noise_std = np.sqrt(noise_power)  # Noise standard deviation is the square root of power
    noise = np.random.normal(scale=noise_std, size=image.shape)  # Generate Gaussian noise
    noisy_image = image + noise  # Add the noise to the signal
    noisy_image = np.clip(noisy_image, 0, 1)  # Clip values to ensure the image range stays valid
    return noisy_image

# Function to add noise to an image
def add_noise_amplitude(image, snr_db):
    snr_linear = 10 ** (snr_db / 20)
    signal_std = np.std(image)
    noise_std = signal_std / snr_linear
    noise = np.random.normal(scale=noise_std, size=image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image



def rstcanet_demosaic(raw_slice, model, device):
    img_input = np.expand_dims(raw_slice, axis=-1)
    img_input_tensor = torch.from_numpy(np.ascontiguousarray(img_input)).permute(2, 0, 1).float().unsqueeze(0)
    img_input_tensor = img_input_tensor.to(device)
    img_output = model(img_input_tensor)
    img_output = img_output.data.squeeze().float().clamp_(0.0, 1.0).cpu().numpy()
    if img_output.ndim == 3:
        img_output = np.transpose(img_output, (1, 2, 0))
    return img_output
    
    

# Function to apply Gaussian or uniform blur in PyTorch
def apply_blur_torch(image, blur_type, kernel_size):
    padding = kernel_size // 2
    if blur_type == 'gaussian':
        sigma = kernel_size / 3
        x = torch.arange(-padding, padding + 1, device=image.device, dtype=image.dtype)
        gaussian_kernel = torch.exp(-0.5 * (x / sigma)**2)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel.view(1, 1, -1) * gaussian_kernel.view(1, -1, 1)
        gaussian_kernel = gaussian_kernel.expand(image.shape[1], 1, kernel_size, kernel_size)

        image = F.pad(image, (padding, padding, padding, padding), mode='reflect')
        image = F.conv2d(image, gaussian_kernel, groups=image.shape[1])
    elif blur_type == 'uniform':
        uniform_kernel = torch.ones((image.shape[1], 1, kernel_size, kernel_size), device=image.device, dtype=image.dtype) / (kernel_size ** 2)
        image = F.pad(image, (padding, padding, padding, padding), mode='reflect')
        image = F.conv2d(image, uniform_kernel, groups=image.shape[1])
    else:
        raise ValueError("Unsupported blur type")
    
    return image

