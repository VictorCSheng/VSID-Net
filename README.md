# VSID-Net

VSID-Net is a denoising method especially for scanning electron microscope (SEM) images. The method is based on the idea of variance stabilization and the deep learning method, according to the characteristics of SEM image noise. More detail can be found in the following paper:

Denoising of scanning electron microscope images for ultrastructure enhancement

## Table of Contents

- [Dependencies](#Dependencies)
- [Instructions for Use](#Instructions-for-Use)
- [Examples and Comparison Results](#Examples-and-Comparison-Results)
- [Contributing](#Contributing)

## Dependencies

Our method was trained on the Pytorch deep learning framework and TensorFlow  backend. 
The required libraries are as follows. 

python3.8, numpy, tifffile, skimage, torch, torchvision, tqdm

If you don't have some of these libraries, you can install them using pip or another package manager.

## Instructions for Use

If you just want to test our method, you can use "./source/denoise_demo.py".

If you want to retrain our network, please check and run "./source/train.py". 

If you want to use our training parameters directly, please run "./source/predict.py".

For the denoising of large-scale SEM images, you may need to apply "./source/batchbigimgdenoise.py".

## Examples and Comparison Results

Here are some examples of denoising SEM images using different denoising algorithms. 
Our method has a good balance between denoising and over-smoothing. And intuitively, our method has also achieved the best results

![Denoising results](https://github.com/VictorCSheng/VSID-Net/raw/main/paper_image/results.png)

## Contributing
Please refer to the paper "Denoising of scanning electron microscope images for ultrastructure enhancement".



