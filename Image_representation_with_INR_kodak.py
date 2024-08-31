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


# Initialize parser
parser = argparse.ArgumentParser(description='INRID2024')

# Shared Parameters
parser.add_argument('--input_dir', type=str, default='kodak', help='Input directory containing images')
parser.add_argument('--output_dir', type=str, default='output/Image_representation_with_INR/', help='Output directory to save results')
parser.add_argument('--inr_models', nargs='+', default=['gauss', 'relu', 'siren', 'wire', 'ffn', 'incode'], help='List of INR models to use')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--using_schedular', type=bool, default=True, help='Whether to use scheduler')
parser.add_argument('--scheduler_b', type=float, default=0.1, help='Learning rate scheduler')
parser.add_argument('--maxpoints', type=int, default=128 * 128, help='Batch size')
parser.add_argument('--steps_til_summary', type=int, default=100, help='Number of steps till summary visualization')
# INCODE Parameters
parser.add_argument('--a_coef', type=float, default=0.1993, help='a coefficient')
parser.add_argument('--b_coef', type=float, default=0.0196, help='b coefficient')
parser.add_argument('--c_coef', type=float, default=0.0588, help='c coefficient')
parser.add_argument('--d_coef', type=float, default=0.0269, help='d coefficient')

args = parser.parse_args(args=[])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get list of images
image_paths = [os.path.join('dat/'+args.input_dir, f) for f in os.listdir('dat/'+args.input_dir) if f.endswith('.png') or f.endswith('.tif')]

# Initialize a dictionary to store PSNR results
psnr_results = {}

niters_list = [501, 1001, 2001]
resize_fact_list = [1, 2, 4]

ensure_dir(os.path.join(args.output_dir, args.input_dir))

# Iterate over number of iterations
for niters in niters_list:
    # Initialize sub-dictionary for current `niters`
    args.niters = niters
    psnr_results[niters] = {}

    # Iterate over resize factors
    for resize_fact in resize_fact_list:
        # Initialize sub-dictionary for current `resize_fact`
        psnr_results[niters][resize_fact] = {}

        # Iterate over all images
        for image_path in image_paths:
            # Initialize sub-dictionary for current `image_path`
            print(f'\n\n{image_path} starts...\n\n')
            psnr_results[niters][resize_fact][image_path] = {}

            im = utils.normalize(plt.imread(image_path).astype(np.float32), True)
            im = cv2.resize(im, None, fx=1 / resize_fact, fy=1 / resize_fact, interpolation=cv2.INTER_AREA)
            H, W, _ = im.shape
            base_name = os.path.basename(image_path).split('.')[0]

            # Iterate over all INR models
            for inr_model in args.inr_models:
                # Frequency Encoding
                pos_encode_freq = {'type': 'frequency', 'use_nyquist': True, 'mapping_input': int(max(H, W) / 3)}
                pos_encode_gaus = {'type': 'gaussian', 'scale_B': 10, 'mapping_input': 256}
                pos_encode_no = {'type': None}

                # Model configuration based on INR model
                if inr_model == 'incode':
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
                                               hidden_features=256,
                                               hidden_layers=3,
                                               first_omega_0=30.0,
                                               hidden_omega_0=30.0,
                                               pos_encode_configs=pos_encode_no,
                                               MLP_configs=MLP_configs
                                               ).to(device)
                elif inr_model == 'siren':
                    model = INR(inr_model).run(in_features=2,
                                               out_features=3,
                                               hidden_features=256,
                                               hidden_layers=3,
                                               first_omega_0=30.0,
                                               hidden_omega_0=30.0
                                               ).to(device)
                elif inr_model == 'wire':
                    model = INR(inr_model).run(in_features=2,
                                               out_features=3,
                                               hidden_features=256,
                                               hidden_layers=3,
                                               first_omega_0=30, hidden_omega_0=0, sigma=10.0,
                                               ).to(device)
                elif inr_model == 'relu' or inr_model == 'gauss':
                    model = INR(inr_model).run(in_features=2,
                                               out_features=3,
                                               hidden_features=256,
                                               hidden_layers=3,
                                               pos_encode_configs=pos_encode_no,
                                               ).to(device)
                elif inr_model == 'ffn':
                    model = INR('relu').run(in_features=2,
                                            out_features=3,
                                            hidden_features=256,
                                            hidden_layers=3,
                                            pos_encode_configs=pos_encode_gaus,
                                            ).to(device)

                if inr_model == 'wire':
                    wire_lr = args.lr * min(1, args.maxpoints / (H * W))
                    optim = torch.optim.Adam(lr=wire_lr, params=model.parameters())
                else:
                    optim = torch.optim.Adam(lr=args.lr, params=model.parameters())
                scheduler = lr_scheduler.LambdaLR(optim, lambda x: args.scheduler_b ** min(x / args.niters, 1))

                psnr_values = []
                mse_array = torch.zeros(args.niters, device=device)
                best_loss = torch.tensor(float('inf'))

                coords = utils.get_coords(H, W, dim=2)[None, ...]
                gt = torch.tensor(im).reshape(H * W, 3)[None, ...].to(device)
                rec = torch.zeros_like(gt)

                for step in tqdm(range(args.niters)):
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

                        output_loss = ((model_output - gt[:, b_indices, :]) ** 2).mean()

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
                        psnr = -10 * torch.log10(mse_array[step])
                        psnr_values.append(psnr.item())

                    if args.using_schedular:
                        if inr_model == 'incode' and 30 < step:
                            scheduler.step()
                        else:
                            scheduler.step()

                    imrec = rec[0, ...].reshape(H, W, 3).detach().cpu().numpy()

                    if (mse_array[step] < best_loss) or (step == 0):
                        best_loss = mse_array[step]
                        best_img = imrec
                        best_img = (best_img - best_img.min()) / (best_img.max() - best_img.min())

                # Save best image
                output_img_dir = os.path.join(args.output_dir, args.input_dir, f'niters_{niters}', f'resize_{resize_fact}', base_name,
                                              inr_model)
                ensure_dir(output_img_dir)
                plt.imsave(os.path.join(output_img_dir, f'best_{base_name}_{inr_model}.png'), best_img)

                # Record the best PSNR
                best_psnr = max(psnr_values)
                psnr_results[niters][resize_fact][image_path][inr_model] = best_psnr
                print(f'{image_path} with {inr_model} - Max PSNR: {best_psnr:.4f}')

# Save PSNR results
output_txt_path = os.path.join(args.output_dir, args.input_dir, 'psnr_results.txt')
with open(output_txt_path, 'w') as f:
    header = 'Niters\tResize_Factor\tImage\tINR_Model\tPSNR\n'
    f.write(header)

    for niters, resize_dict in psnr_results.items():
        for resize_fact, image_dict in resize_dict.items():
            for image_path, inr_dict in image_dict.items():
                base_name = os.path.basename(image_path).split('.')[0]
                for inr_model, psnr in inr_dict.items():
                    psnr_line = f"{niters}\t{resize_fact}\t{base_name}\t{inr_model}\t{psnr:.4f}\n"
                    f.write(psnr_line)

print('PSNR results saved to:', output_txt_path)

