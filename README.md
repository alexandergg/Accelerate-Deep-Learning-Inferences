---
page_type: sample
languages:
- python
products:
- Tensorflow 2.0
- TF-Lite
- NVIDIA TF2.0 - TF-TRT
description: "TensorFlow model optimization using TensorRT and TF-Lite"
---

# High performance inference with Tensorflow 2.0 and TF-TensorRT Integration

![](https://cdn-images-1.medium.com/max/1600/0*KvvRkxZfnZiROVqj)

Thanks to TensorFlow 2.0 we can obtain high performance for deep learning inference through a simple API. This repository use simple examples to show you how to optimize an app using TensorRT or TF-Lite with the new Keras APIs in TensorFlow 2.0. This notebooks will show you tips and tricks to get the highest performance possible on GPUs and detail examples of how to debug and profile apps using tools by NVIDIA and TensorFlow. You’ll walk away with an overview and resources to get started, and if you’re already familiar with TensorFlow, you’ll get tips on how to get the most out of your application.

By running this project, you will have the opportunity to work with Tensorflow optimization toolkit and NVIDIA SDK for high-performance deep learning inference. It includes a deep learning inference optimizer and runtime that delivers low latency and high-throughput for deep learning inference applications.

|Technology|
|----------|
|Tensorflow 2.0 |
|TF-Lite Backend |
|NVIDIA TF2.0 - TF-TRT |

## Virtual environment to execute tflite notebook

### Ananconda and Jupyter Notebook Local

To create the virual environment, we need to have anaconda installed in our computer. It can be downloaded in this [link](https://www.anaconda.com/download/)

- Instalation: https://www.anaconda.com/distribution/
- Conda commands to create local env by environment.yml: ```conda env create -f environment.yml```
- Set conda env into jupyter notebook: ```python -m ipykernel install --user --name <environment_name> --display-name "Python (<environment_name>)"```

Once the environment is created, to activate it:

`conda activate <environment-name>`

To deactivate the environment:

`conda deactivate <environment-name>`

## Virtual environment to execute tensorrt notebook

Open the notebook at https://colab.research.google.com/

Why **Colab**? Actually, Tensorflow TF-TRT does not support on Windows 10 also we can't execute it in WSL because we dont have access using WSL to a GPU yet.
If we use Colab, we have a full Linux environment and completely access to a free virtual machine with GPU!

If you have Linux OS, use tensorflow-gpu>=2.0. (Support TensorRT)

## TF-Lite Quantized Int8 Model to Edge TPU

Open the notebook at https://colab.research.google.com/

In this notebook you can compile your TF-Lite int8 quantized model into TPU Compiler. Specially, this step is to obtain a model compiled to use, for example into Coral Dev Board or any device with [TPU](https://coral.ai/docs/edgetpu/compiler/) (**Tensor Preprocessor Unit**) 


### References

Thanks to **Tensorflow** Communnity, keep growing!!

- [Tensorflow 2.0](https://www.tensorflow.org/learn)
- [TF Optimization Toolkit](https://www.tensorflow.org/lite/guide/get_started)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)