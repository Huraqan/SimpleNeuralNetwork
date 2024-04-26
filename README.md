# SimpleNeuralNetwork

This project is a showcase on how to build and train a simple neural network on images of handwritten digits. It is coded in Python 3.12 using basic libraries like Numpy and Pandas.

# Usage

Explore the concept of neural networks by getting your hands dirty.
See the code and maths for the feed-forward and back-propagation.
Tinker with settings to get improved results. Expand upon the project by implementing convolution, a fantansic way to get more valuable information encoded into each pixel, allowing for the network to better differentiate certain patterns.

# Installation

Use `pip install -r requirements.txt` to get necessary libs.

# Getting started

You can start playing around with `playground.ipynb`, a notebook where you can freely train and inspect your current model.

If you want to do some analysis on the validation results, a notebook has been prepared `analysis.ipynb`

The network itself is composed of two files:
- SimpleNeuralNetwork.py
- Layer.py

This network has been set up to accomodate two output shapes: (10, 1) and (11, 1). The first ten elements correspond to each digit. The eleventh element represents the "noise" category/target.

I have noticed training the model to recognize noise will improve training speed, but also generel accuracy. At least for the first epochs. More epochs seem to bring results of noise and noiseless models much closer together.

However, I do find it fascinating that the model is in fact better at recognizing random noise than it is at distinguishing digits.