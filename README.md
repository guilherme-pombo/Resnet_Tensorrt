# Resnet_TensorRT
Small wrapper around Keras' Resnets to transform them into quick UFF models that can use Nvidia's TensorRT

# Purpose

Using Keras directly with high-end GPUs such as the V100 results in relatively poor utilisation.
If you're renting a good GPU AWS machine such as the p3.2xlarge, you probably want to make the most out of your money.
Luckily Nvidia developed TensorRT which is a high-performance deep learning inference optimizer and runtime that delivers 
low latency, high-throughput inference for deep learning applications.

https://developer.nvidia.com/tensorrt

Many times, doing inference with a model is the most expensive part of a machine learning pipeline.
Hence, it makes sense to try and optimise inference as much as possible.
In this repo I provide an easy and small wrapper (150 loc) that will allow you to use your Keras' Resnet models with Nvidia's TensorRT

# Installing Nvidia TensorRT

Please check Nvidia's page for a really good guide on doing this:

http://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html


# Quick example

```
import numpy as np

from keras.applications import resnet50
from tensorrt_wrapper import TensorrtWrapper, convert_keras_to_uff_model

# load in the original Keras model from disk -- example with imagenet weights
model = resnet50.ResNet50(include_top=True, weights='imagenet')
model.load_weights('resnet50_example.h5')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# convert it to an UFF model
model_input_name, model_output_name = convert_keras_to_uff_model(model, 'uff_resnet_50.uff')

# now use the UFF model with Nvidia's Tensorrt library to speed up predictions
tw = TensorrtWrapper('uff_resnet_50.uff',
                     model_input_name=model_input_name,
                     model_output_name=model_output_name)

# generate fake images
images = []
for _ in range(128):
    test_image = np.random.rand(224, 224, 3)
    images.append(test_image)
images = np.array(images)
images = np.transpose(images, axes=(0, 3, 1, 2)).astype(np.float32)
images = images.copy(order='C')
tw.run_prediction(images)
print(tw.output)
```

# Results

I've been using this to perform multi-class inference on Petabytes of imagery and have been getting around a 20x speedup
 when compared to using just Keras -> Resulting in really good $$$$ savings
 


