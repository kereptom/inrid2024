# INRID: Implicit Neural Representation for Image Demosaicing

## Abstract

This repository contains the implementation of INRID (Implicit Neural Representation for Image Demosaicing), a novel approach that leverages implicit neural representations (INRs) to enhance image demosaicing algorithms. INRID employs a multi-layer perceptron (MLP) to implicitly represent RGB images, utilizing both the original Bayer measurements and an initial demosaicing estimate from traditional methods to achieve superior results. A key innovation is the introduction of a Bayer loss function, which computes the mean squared error relative to the original sensor data. This approach not only improves the final image quality but also demonstrates the capacity to correct state-of-the-art demosaicing algorithms when input data diverge from the training distribution, such as in cases of blur or noise.

## Repository Contents

### Folders

- dat/: Contains datasets used for training and testing, such as the Kodak and McMaster datasets.

- models/: Contains pretrained demosaicing models.

- modules/: Contains utility modules, helper functions used across different scripts  and architectures of the implicit neural networks used in the experiments. This includes implementations of different INR architectures like SIREN, INCODE, WIRE, etc.


### Usage

#### Image Representation with INR

To run the image representation experiments where the INR models are trained to fit original images:

   python Image_representation_with_INR.py

#### Image Demosaicing with INR

To perform naive INR demosaicing by overfitting to Bayer measurements:

   python Image_demosaicing_with_INR.py

#### Boosting Traditional Demosaicing Methods

To enhance traditional demosaicing methods like Malvar and Menon:

   python Boosting_Image_demosaicing_with_INR.py

#### Boosting RSTCANet Demosaicing Method

To enhance the RSTCANet demosaicing method using INRID:

   python Boosting_Image_demosaicing_with_INR_RSTCANet.py

#### Joint Demosaicing and Denoising

To run joint demosaicing and denoising experiments:

   python Joint_Demosaicing_and_Denoising.py

#### Joint Demosaicing and Deblurring

To run joint demosaicing and deblurring experiments:

   python Joint_Demosaicing_and_Debluring.py

### Results

The scripts will output reconstructed images and performance metrics such as PSNR and SSIM. Results can be saved to the results/ directory or any specified output path.


## Experimental Details

- INR Architectures: The experiments utilize different INR architectures including SIREN, INCODE, WIRE, etc. Each architecture can be selected and configured within the scripts.

- Loss Functions: The implementation includes the Bayer loss and complementary loss functions as described in the paper.

- Optimization: The models are trained using the Adam optimizer with specific learning rates and parameters set in the scripts.

- Evaluation Metrics: The performance is evaluated using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM).
