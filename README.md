# Behavioral Cloning
## Goal
This project uses a Convolutional Neural Network model to attemp to learn how to drive a car in a simulator based on the data of a human driving the car.

## Simulator

The car simulator used to gather training data is made by Udacity for their [Self-Driving Car Nanodegree](https://www.udacity.com/drive) program, download it here:

   [MacOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983385_beta-simulator-mac/beta-simulator-mac.zip)   [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983318_beta-simulator-windows/beta-simulator-windows.zip)   [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983558_beta-simulator-linux/beta-simulator-linux.zip) 

## Running training

To run training on the model, use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) in order to train on the GPU,
use the following commands:

```sh
git clone https://github.com/Charles-Catta/Behavioral-Cloning.git

cd Behavioral-Cloning

nvidia-docker run -it --rm -v `pwd`:/workdir madhorse/behavioral-cloning python3 model.py
```

## Model Architecture

![Model Architecture](img/model.png)

The model architecture for this project is based on Nvidia's paper on [_End to end learning for self-driving cars_](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)



