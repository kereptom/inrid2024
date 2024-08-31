import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from tqdm.notebook import tqdm

import torch
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler

from modules import utils
from modules.models import INR
from myutils import *

from colour_demosaicing import (
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
)

from PIL import Image


# Initialize parser
parser = argparse.ArgumentParser(description='INRID2024_Boosting_Image_Demo_INR')

# Shared Parameters
parser.add_argument('--input_dir', type=str, default='kodak1', help='Input directory containing images')
parser.add_argument('--output_dir', type=str, default='output/Boosting_Image_demosaicing_with_INR/', help='Output directory to save results')
parser.add_argument('--inr_models', nargs='+', default=['siren', 'incode'], help='List of INR models to use')
parser.add_argument('--niters_list', nargs='+',  type=int,  default=[2001, 6001, 8001, 10001], help='List of number of iterations')
parser.add_argument('--resize_fact_list', nargs='+',  type=int, default=[4], help='List of resize factors')

parser.add_argument('--lr_gauss', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lr_relu', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lr_siren', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lr_wire', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lr_ffn', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lr_incode', type=float, default=1e-4, help='Learning rate')

parser.add_argument('--hidden_layers', type=int, default=5, help='Number of hidden layers')
parser.add_argument('--hidden_features', type=int, default=256, help='Number of hidden features')
parser.add_argument('--using_schedular', type=bool, default=True, help='Whether to use scheduler')
parser.add_argument('--scheduler_b', type=float, default=0.1, help='Learning rate scheduler')
parser.add_argument('--maxpoints', type=int, default=128 * 128, help='Batch size')
parser.add_argument('--steps_til_summary', type=int, default=100, help='Number of steps till summary visualization')
# INCODE Parameters
parser.add_argument('--a_coef', type=float, default=0.1993, help='a coefficient')
parser.add_argument('--b_coef', type=float, default=0.0196, help='b coefficient')
parser.add_argument('--c_coef', type=float, default=0.0588, help='c coefficient')
parser.add_argument('--d_coef', type=float, default=0.0269, help='d coefficient')

parser.add_argument('--alpha_list', nargs='+', type=float, default=[0.1,1,60,100,1000], help='Weighting Bayer loss')
parser.add_argument('--beta', type=float, default=1, help='Weighting Demo loss')
parser.add_argument('--demo_method_list', nargs='+', type=str, default=['Nearest','Bilinear', 'Malvar', 'Menon'], help='Demosaicing method')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_paths = [os.path.join('dat/'+args.input_dir, f) for f in os.listdir('dat/'+args.input_dir) if f.endswith('.png') or f.endswith('.tif')]
psnr_results = {}
ssim_results = {}

demosaic_methods = {
    'Nearest': nearest_neighbor_demosaic,
    'Bilinear': demosaicing_CFA_Bayer_bilinear,
    'Malvar': demosaicing_CFA_Bayer_Malvar2004,
    'Menon': demosaicing_CFA_Bayer_Menon2007
}

for alpha in args.alpha_list:
    psnr_results[alpha] = {}
    ssim_results[alpha] = {}  
    
    for demo_method in args.demo_method_list:
        psnr_results[alpha][demo_method] = {}
        ssim_results[alpha][demo_method] = {}     
            
        for niters in args.niters_list:
            psnr_results[alpha][demo_method][niters] = {}
            ssim_results[alpha][demo_method][niters] = {}
    
            for resize_fact in args.resize_fact_list:
                psnr_results[alpha][demo_method][niters][resize_fact] = {}
                ssim_results[alpha][demo_method][niters][resize_fact] = {}
                
                for image_path in image_paths:
                    print(f'\n\n{image_path} starts...\n\n')
                    psnr_results[alpha][demo_method][niters][resize_fact][image_path] = {}
                    ssim_results[alpha][demo_method][niters][resize_fact][image_path] = {}

                    im = utils.normalize(plt.imread(image_path).astype(np.float32), True)
                    im = cv2.resize(im, None, fx=1 / resize_fact, fy=1 / resize_fact, interpolation=cv2.INTER_AREA)
                    H, W, _ = im.shape
                    base_name = os.path.basename(image_path).split('.')[0]
                    bayer_mask = create_bayer_mask(H, W, device)
                    compl_bayer_mask = create_complementary_bayer_mask(H, W, device)
                    im_bayer2 = rgb2RGGB(im)
                    demo_orig = demosaic_methods[demo_method](im_bayer2, 'RGGB')

                    for inr_model in args.inr_models:
                        pos_encode_gaus = {'type': 'gaussian', 'scale_B': 10, 'mapping_input': args.hidden_features}
                        pos_encode_no = {'type': None}

                        if inr_model == 'incode':
                            args.lr = args.lr_incode 
                            MLP_configs = {'task': 'image',
                                           'model': 'resnet34',
                                           'truncated_layer': 5,
                                           'in_channels': 64,
                                           'hidden_channels': [64, 32, 4],
                                           'mlp_bias': 0.3120,
                                           'activation_layer': nn.SiLU,
                                           'GT': torch.tensor(im).to(device)[None, ...].permute(0, 3, 1, 2)
                                           }

                            model = INR(inr_model).run(in_features=2,
                                                       out_features=3,
                                                       hidden_features=args.hidden_features,
                                                       hidden_layers=args.hidden_layers,
                                                       first_omega_0=30.0,
                                                       hidden_omega_0=30.0,
                                                       pos_encode_configs=pos_encode_no,
                                                       MLP_configs=MLP_configs
                                                       ).to(device)
                        elif inr_model == 'siren':
                            args.lr = args.lr_siren 
                            model = INR(inr_model).run(in_features=2,
                                                       out_features=3,
                                                       hidden_features=args.hidden_features,
                                                       hidden_layers=args.hidden_layers,
                                                       first_omega_0=30.0,
                                                       hidden_omega_0=30.0
                                                       ).to(device)
                        elif inr_model == 'wire':
                            args.lr = args.lr_wire 
                            model = INR(inr_model).run(in_features=2,
                                                       out_features=3,
                                                       hidden_features=args.hidden_features,
                                                       hidden_layers=args.hidden_layers,
                                                       first_omega_0=30, hidden_omega_0=10, sigma=10.0,
                                                       ).to(device)
                        elif inr_model == 'relu':
                            args.lr = args.lr_relu 
                            model = INR(inr_model).run(in_features=2,
                                                       out_features=3,
                                                       hidden_features=args.hidden_features,
                                                       hidden_layers=args.hidden_layers,
                                                       pos_encode_configs=pos_encode_no,
                                                       ).to(device)
                        elif inr_model == 'gauss':
                            args.lr = args.lr_gauss 
                            model = INR(inr_model).run(in_features=2,
                                                       out_features=3,
                                                       hidden_features=args.hidden_features,
                                                       hidden_layers=args.hidden_layers,
                                                       pos_encode_configs=pos_encode_no,
                                                       ).to(device)
                        elif inr_model == 'ffn':
                            args.lr = args.lr_ffn 
                            model = INR('relu').run(in_features=2,
                                                    out_features=3,
                                                    hidden_features=args.hidden_features,
                                                    hidden_layers=args.hidden_layers,
                                                    pos_encode_configs=pos_encode_gaus,
                                                    ).to(device)
                                                    
                                 

                        if inr_model == 'wire':
                            wire_lr = args.lr * min(1, args.maxpoints / (H * W))
                            optim = torch.optim.Adam(lr=wire_lr, params=model.parameters())
                        else:
                            optim = torch.optim.Adam(lr=args.lr, params=model.parameters())
                        scheduler = lr_scheduler.LambdaLR(optim, lambda x: args.scheduler_b ** min(x / niters, 1))  
                        
                        

                        psnr_values = []
                        ssim_values = []
                        mse_array = torch.zeros(niters, device=device)
                        best_loss = torch.tensor(float('inf'))

                        coords = utils.get_coords(H, W, dim=2)[None, ...]
                        gt = torch.tensor(im).reshape(H * W, 3)[None, ...].to(device)
                        gt_bayer = gt * bayer_mask
                        dm = torch.tensor(demo_orig).to(device).reshape(H * W, 3)[None, ...]
                        dm_compl = dm * compl_bayer_mask
                        rec = torch.zeros_like(gt)

                        for step in tqdm(range(niters)):
                            indices = torch.randperm(H * W)
                            for b_idx in range(0, H * W, args.maxpoints):
                                b_indices = indices[b_idx:min(H * W, b_idx + args.maxpoints)]
                                b_coords = coords[:, b_indices, ...].to(device)
                                b_indices = b_indices.to(device)

                                if inr_model == 'incode':
                                    model_output, coef = model(b_coords)
                                else:
                                    model_output = model(b_coords)

                                with torch.no_grad():
                                    rec[:, b_indices, :] = model_output

                                model_output_bayer = model_output * bayer_mask[:, b_indices, :]
                                model_output_compl = model_output * compl_bayer_mask[:, b_indices, :]

                                L_Bayer_loss = ((model_output_bayer - gt_bayer[:, b_indices, :]) ** 2).mean()
                                L_Demo_loss = ((model_output_compl - dm_compl[:, b_indices, :]) ** 2).mean()

                                output_loss = alpha * L_Bayer_loss + args.beta * L_Demo_loss

                                if inr_model == 'incode':
                                    a_coef, b_coef, c_coef, d_coef = coef[0]
                                    reg_loss = args.a_coef * torch.relu(-a_coef) + \
                                               args.b_coef * torch.relu(-b_coef) + \
                                               args.c_coef * torch.relu(-c_coef) + \
                                               args.d_coef * torch.relu(-d_coef)
                                    loss = output_loss + reg_loss
                                else:
                                    loss = output_loss

                                optim.zero_grad()
                                loss.backward()
                                optim.step()

                            with torch.no_grad():
                                mse_array[step] = ((gt - rec) ** 2).mean().item()
                                psnr = calculate_psnr(gt.reshape(H, W, 3).cpu().numpy()* 255, rec.reshape(H, W, 3).cpu().numpy() * 255)
                                ssim = calculate_ssim(gt.reshape(H, W, 3).cpu().numpy()* 255, rec.reshape(H, W, 3).cpu().numpy() * 255)
                                psnr_values.append(psnr)
                                ssim_values.append(ssim)

                            if args.using_schedular:
                                if inr_model == 'incode' and 30 < step:
                                    scheduler.step()
                                else:
                                    scheduler.step()

                            if (mse_array[step] < best_loss) or (step == 0):
                                best_loss = mse_array[step]
                                imrec_full = rec.reshape(H, W, 3).cpu().numpy() * 255
                                
                        dm_full = dm.reshape(H, W, 3).cpu().numpy() * 255
                        dm_base = (dm_compl + gt_bayer).reshape(H, W, 3).cpu().numpy() * 255

                                                   
                        raw_slice = (im_bayer2 * 255).astype(np.uint8)
                        im_rgb2 = postprocess_image_simple(im)
                        imrec_full2 = postprocess_image_simple(imrec_full)
                        dm_full2 = postprocess_image_simple(dm_full)
                        dm_base2 = postprocess_image_simple(dm_base)
                        dm_base2_clipped = np.clip(((((dm_base2.astype(np.float32) - dm_base2.mean()) / dm_base2.std()) * im_rgb2.std()) + im_rgb2.mean()), 0, 255).astype(np.uint8)
                        dm_full2_clipped = np.clip(((((dm_full2.astype(np.float32) - dm_full2.mean()) / dm_full2.std()) * im_rgb2.std()) + im_rgb2.mean()), 0, 255).astype(np.uint8)
                        imrec_full2_clipped = np.clip(((((imrec_full2.astype(np.float32) - imrec_full2.mean()) / imrec_full2.std()) * im_rgb2.std()) + im_rgb2.mean()), 0, 255).astype(np.uint8)

                        output_img_dir = os.path.join(args.output_dir, args.input_dir,f'alpha_{alpha}',f'demo_{demo_method}', f'niters_{niters}', f'resize_{resize_fact}', base_name,
                                                      inr_model)
                        ensure_dir(output_img_dir)

                        Image.fromarray(raw_slice).save(os.path.join(output_img_dir, 'Bayer.png'), format='PNG')
                        Image.fromarray(im_rgb2).save(os.path.join(output_img_dir, 'Original.png'), format='PNG')
                        Image.fromarray(dm_full2_clipped).save(os.path.join(output_img_dir, f'{demo_method}_Demosaiced.png'), format='PNG')
                        Image.fromarray(imrec_full2_clipped).save(os.path.join(output_img_dir, f'{demo_method}_INRID.png'), format='PNG')
                        Image.fromarray(dm_base2_clipped).save(os.path.join(output_img_dir, f'{demo_method}_Demosaiced_plus_bayer.png'), format='PNG')                            

                        # Record the best PSNR
                        best_psnr = max(psnr_values)
                        best_ssim = max(ssim_values)
                        psnr_results[alpha][demo_method][niters][resize_fact][image_path][inr_model] = best_psnr
                        ssim_results[alpha][demo_method][niters][resize_fact][image_path][inr_model] = best_ssim
                        print(f'{image_path} with {inr_model} - Max PSNR: {best_psnr:.4f}')
                        dmpsnr = calculate_psnr(im* 255, demo_orig * 255)
                        print(f'Demosaicing PSNR: {dmpsnr:.4f}')
                            

# Save PSNR and SSIM results
output_txt_path = os.path.join(args.output_dir, args.input_dir, 'psnr_ssim_results.txt')
with open(output_txt_path, 'w') as f:
    header = 'Alpha\tDemosaic_Method\tNiters\tResize_Factor\tImage\tINR_Model\tPSNR\tSSIM\n'
    f.write(header)

    for alpha, demo_dict in psnr_results.items():
        for demo_method, niters_dict in demo_dict.items():
            for niters, resize_dict in niters_dict.items():
                for resize_fact, image_dict in resize_dict.items():
                    for image_path, inr_dict in image_dict.items():
                        base_name = os.path.basename(image_path).split('.')[0]
                        for inr_model, psnr in inr_dict.items():
                            ssim = ssim_results[alpha][demo_method][niters][resize_fact][image_path][inr_model]
                            result_line = f"{alpha}\t{demo_method}\t{niters}\t{resize_fact}\t{base_name}\t{inr_model}\t{psnr:.4f}\t{ssim:.4f}\n"
                            f.write(result_line)

print('PSNR and SSIM results saved to:', output_txt_path)

# Calculate and save average PSNR and SSIM results
average_psnr_results = {}
average_ssim_results = {}
output_avg_txt_path = os.path.join(args.output_dir, args.input_dir, 'psnr_ssim_average_results.txt')

for alpha, demo_dict in psnr_results.items():
    average_psnr_results[alpha] = {}
    average_ssim_results[alpha] = {}
    
    for demo_method, niters_dict in demo_dict.items():
        average_psnr_results[alpha][demo_method] = {}
        average_ssim_results[alpha][demo_method] = {}
        
        for niters, resize_dict in niters_dict.items():
            average_psnr_results[alpha][demo_method][niters] = {}
            average_ssim_results[alpha][demo_method][niters] = {}
            
            for resize_fact, image_dict in resize_dict.items():
                if resize_fact not in average_psnr_results[alpha][demo_method][niters]:
                    average_psnr_results[alpha][demo_method][niters][resize_fact] = {}
                    average_ssim_results[alpha][demo_method][niters][resize_fact] = {}

                for inr_model in args.inr_models:
                    psnr_list = []
                    ssim_list = []
                    for image_path in image_dict.keys():
                        if inr_model in psnr_results[alpha][demo_method][niters][resize_fact][image_path]:
                            psnr_list.append(psnr_results[alpha][demo_method][niters][resize_fact][image_path][inr_model])
                            ssim_list.append(ssim_results[alpha][demo_method][niters][resize_fact][image_path][inr_model])
                    
                    if psnr_list:
                        average_psnr = np.mean(psnr_list)
                        average_ssim = np.mean(ssim_list)
                        average_psnr_results[alpha][demo_method][niters][resize_fact][inr_model] = average_psnr
                        average_ssim_results[alpha][demo_method][niters][resize_fact][inr_model] = average_ssim

with open(output_avg_txt_path, 'w') as f:
    header = 'Alpha\tDemosaic_Method\tNiters\tResize_Factor\tINR_Model\tAverage_PSNR\tAverage_SSIM\n'
    f.write(header)

    for alpha, demo_dict in average_psnr_results.items():
        for demo_method, niters_dict in demo_dict.items():
            for niters, resize_dict in niters_dict.items():
                for resize_fact, inr_dict in resize_dict.items():
                    for inr_model, avg_psnr in inr_dict.items():
                        avg_ssim = average_ssim_results[alpha][demo_method][niters][resize_fact][inr_model]
                        avg_result_line = f"{alpha}\t{demo_method}\t{niters}\t{resize_fact}\t{inr_model}\t{avg_psnr:.4f}\t{avg_ssim:.4f}\n"
                        f.write(avg_result_line)

print('Average PSNR and SSIM results saved to:', output_avg_txt_path)

# Save all actual argument parameters to setup.txt
setup_txt_path = os.path.join(args.output_dir, args.input_dir, 'setup.txt')

# Ensure the output directory exists
ensure_dir(os.path.dirname(setup_txt_path))

with open(setup_txt_path, 'w') as f:
    for arg in vars(args):
        f.write(f'{arg}: {getattr(args, arg)}\n')

print('Setup parameters saved to:', setup_txt_path)

# Create or open a text file to save the PSNR and SSIM results for each image and demosaic method
output_baseline_txt_path = os.path.join(args.output_dir, args.input_dir, 'psnr_ssim_baseline_results.txt')
output_avg_baseline_txt_path = os.path.join(args.output_dir, args.input_dir, 'psnr_ssim_average_baseline_results.txt')

baseline_psnr_dict = {method: [] for method in args.demo_method_list}
baseline_ssim_dict = {method: [] for method in args.demo_method_list}

with open(output_baseline_txt_path, 'w') as baseline_file:
    header = 'Image\tDemosaic_Method\tPSNR\tSSIM\n'
    baseline_file.write(header)
    
    # Loop through all combinations of demosaic methods and images to log PSNR and SSIM
    for demo_method in args.demo_method_list:
        for image_path in image_paths:
            # Load the image and perform necessary preprocessing
            im = utils.normalize(plt.imread(image_path).astype(np.float32), True)
            im = cv2.resize(im, None, fx=1 / resize_fact, fy=1 / resize_fact, interpolation=cv2.INTER_AREA)
            base_name = os.path.basename(image_path).split('.')[0]
            im_bayer2 = rgb2RGGB(im)
            demo_orig = demosaic_methods[demo_method](im_bayer2, 'RGGB')
            
            # Calculate PSNR and SSIM between original image and demosaiced image
            dmpsnr = calculate_psnr(im * 255, demo_orig * 255)
            dmssim = calculate_ssim(im * 255, demo_orig * 255)
            
            # Append to the baseline lists for averaging later
            baseline_psnr_dict[demo_method].append(dmpsnr)
            baseline_ssim_dict[demo_method].append(dmssim)
            
            # Print the results for this image and demosaic method
            print(f'{image_path} with {demo_method} - PSNR: {dmpsnr:.4f}, SSIM: {dmssim:.4f}')
            
            # Write the results to the text file
            baseline_file.write(f"{base_name}\t{demo_method}\t{dmpsnr:.4f}\t{dmssim:.4f}\n")

# Calculate and save average PSNR and SSIM for each demosaicing method
with open(output_avg_baseline_txt_path, 'w') as avg_baseline_file:
    header = 'Demosaic_Method\tAverage_PSNR\tAverage_SSIM\n'
    avg_baseline_file.write(header)

    for demo_method in args.demo_method_list:
        average_psnr = np.mean(baseline_psnr_dict[demo_method])
        average_ssim = np.mean(baseline_ssim_dict[demo_method])
        avg_baseline_file.write(f"{demo_method}\t{average_psnr:.4f}\t{average_ssim:.4f}\n")

print('Baseline average PSNR and SSIM results saved to:', output_avg_baseline_txt_path)

