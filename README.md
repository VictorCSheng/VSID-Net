# VSID-Net

VSID net is a denoising method especially for scanning electron microscope (SEM) images. The method is based on the idea of variance stabilization and the deep learning method, according to the characteristics of SEM image noise. More detail can be found in the following paper:

Denoising of scanning electron microscope images for ultrastructure enhancement

## Table of Contents

- [Dependencies](#Dependencies)
- [Instructions for Use](#Instructions-for-Use)
- [Examples and Comparison Results](#Examples-and-Comparison-Results)
- [Contributing](#Contributing)

## Dependencies

Our method was trained on the Pytorch deep learning framework and TensorFlow  backend. 
The required libraries are as follows. 

python3, numpy, tifffile, skimage, torch, torchvision, tqdm

If you don't have some of these libraries, you can install them using pip or another package manager.

## Instructions for Use

If you want to retrain our network, please check and run "train.py". 
If you want to use our training parameters directly, please run "predict.py".
For the denoising of large-scale SEM images, you may need to apply "batchbigimgdenoise.py", due to the limitation of computer memory.

## Examples and Comparison Results

Here are some examples of denoising SEM images using different denoising algorithms. 
Our method has a good balance between denoising and over-smoothing and achieved the best denoising effect from the intuitive experience of denoising images.

![Denoising results](https://github.com/VictorCSheng/VSID-Net/tree/main/example_image/results.PNG)

## Contributing
Please refer to the paper "Denoising of scanning electron microscope images for ultrastructure enhancement".



