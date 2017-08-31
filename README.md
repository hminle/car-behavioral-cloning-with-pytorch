# Car Behavioral Cloning using Pytorch

## This is a work in progress

## Overview

I create this project after watching Siraj's video about how to simulate a self-driving car.
Instead using Keras to build a model, I want to using Pytorch for the model and training.

The credis for this code goto [naokishibuya](https://github.com/naokishibuya).

## Challenges:

- Building a model in Pytorch is not as straighforward as in Keras. You need to understand the framework and how it processes data first.
- Need to create a Dataloader for your own data.
- Re-use as much code as possible.

## What has been done so far

- Created the Dataloader for car dataset
- Created the CarModel for training with the dataset. The model is based on The [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).
![NIVIDIA Model](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)
- Modified drive.py so that it can load the Pytorch model.

## Notes

- Training the neural network

![Training the model](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/training-624x291.png)

- After trained, the network is able to generate steering commands from the video images of a single center camera.

![generate steering](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/inference-624x132.png)

- Testing the model with the Simulator

![Testing the model](https://camo.githubusercontent.com/e225b508bec2b7d4792856f1881ad77abc5fac7b/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313434302f312a6e6c7573615f664335426e73676e5750466e6f7637512e74696666)


## Dependencies

- You can install all dependencies by running conda:

```bash
conda env create -f environments.yml
```
- Note: I've removed tensorflow in this file
- After that, you need to install [Pytorch](pytorch.org)
- To run up the server, you need to download [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim). (using Autonomous Mode when trying your model)
