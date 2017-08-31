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
- Created the CarModel for training with the dataset
- Modified drive.py so that it can load the Pytorch model.

## Dependencies

- You can install all dependencies by running conda:

```bash
conda env create -f environments.yml
```
- Note: I've removed tensorflow in this file
- After that, you need to install [Pytorch](pytorch.org)
- To run up the server, you need to download [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim). (using Autonomous Mode when trying your model)
