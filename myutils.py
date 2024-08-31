import os
import torch
import cv2
import math
import numpy as np

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
