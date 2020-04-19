import os
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.keras.applications.resnet_v2 import (
        preprocess_input as resnet_v2_preprocess_input,
        decode_predictions as resnet_v2_decode_predictions
)

def load_image(i, target_size=(130, 130)):
    image_path = './test/'+ os.listdir('./test/')[i]
    img = image.load_img(image_path, target_size=target_size)
    
    return (img, image_path)


def get_images(number_of_images, get_one_image=load_image):
    images = []
    
    for i in range(number_of_images):
        images.append(get_one_image(i))

    return images


def batch_input(images):
    batch_size = len(images)
    batched_input = np.zeros((batch_size, 130, 130, 3), dtype=np.float32)
    
    for i in range(batch_size):
        img = images[i][0]
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = resnet_v2_preprocess_input(x)
        batched_input[i, :] = x
        
    batched_input = tf.constant(batched_input)
    return batched_input


def predict_and_benchmark_throughput(batched_input, model, N_warmup_run=50, N_run=500):
    elapsed_time = []
    all_preds = []
    batch_size = batched_input.shape[0]
    
    for i in range(N_warmup_run):
        preds = model.predict(batched_input)

    for i in range(N_run):
        start_time = time.time()
        preds = model.predict(batched_input)
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        all_preds.append(preds)
        
        if i % 50 == 0:
            print('Steps {}-{} average: {:4.1f}ms'.format(i, i+50, (elapsed_time[-50:].mean()) * 1000))
            
    print('Throughput: {:.0f} images/s'.format(N_run * batch_size / elapsed_time.sum()))
    return all_preds


def predict_and_benchmark_throughput_from_saved(batched_input, infer, N_warmup_run=50, N_run=500, model='custom'):
    elapsed_time = []
    all_preds = []
    batch_size = batched_input.shape[0]
    
    for i in range(N_warmup_run):
        labeling = infer(batched_input)

        if model == "quantized":
          preds = labeling
        if model == "custom":
            preds = labeling['dense_1'].numpy()

    for i in range(N_run):
        start_time = time.time()
        labeling = infer(batched_input)
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)

        if model == "quantized":
          preds = labeling
        if model == "custom":
            preds = labeling['dense_1'].numpy()

        all_preds.append(preds)
        
        if i % 50 == 0:
            print('Steps {}-{} average: {:4.1f}ms'.format(i, i+50, (elapsed_time[-50:].mean()) * 1000))
            
    print('Throughput: {:.0f} images/s'.format(N_run * batch_size / elapsed_time.sum()))
    return all_preds


def display_prediction_info(preds, images):
  class_names = ['Parasitized', 'Uninfected']
  for i in range(len(preds)):
    img_decoded_predictions = preds[i]
    predictions = img_decoded_predictions.flatten()
    img, path = images[i]
    print(path)
    for i, predicted in enumerate(predictions):
      if predicted > 0.25:
          result = class_names[1]
      else:
          result = class_names[0]

    plt.figure()
    plt.axis('off') 
    plt.title(result)
    plt.imshow(img)
    plt.show()


def load_tf_saved_model(input_saved_model_dir):
    print('Loading saved model {}...'.format(input_saved_model_dir))
    saved_model_loaded = tf.saved_model.load(input_saved_model_dir, tags=[tag_constants.SERVING])
    return saved_model_loaded


def convert_to_trt_graph_and_save(precision_mode='float32', input_saved_model_dir='malaria_model', calibration_data=''):
    if precision_mode == 'float32':
        precision_mode = trt.TrtPrecisionMode.FP32
        converted_save_suffix = '_TFTRT_FP32'
        
    if precision_mode == 'float16':
        precision_mode = trt.TrtPrecisionMode.FP16
        converted_save_suffix = '_TFTRT_FP16'

    if precision_mode == 'int8':
        precision_mode = trt.TrtPrecisionMode.INT8
        converted_save_suffix = '_TFTRT_INT8'
        
    output_saved_model_dir = input_saved_model_dir + converted_save_suffix
    
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=precision_mode, 
        max_workspace_size_bytes=8000000000
    )

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_saved_model_dir,
        conversion_params=conversion_params
    )

    print('Converting {} to TF-TRT graph precision mode {}...'.format(input_saved_model_dir, precision_mode))
    
    if precision_mode == trt.TrtPrecisionMode.INT8:
        
        def calibration_input_fn():
            yield (calibration_data, )

        converter.convert(calibration_input_fn=calibration_input_fn)   
    else:
        converter.convert()

    print('Saving converted model to {}...'.format(output_saved_model_dir))
    converter.save(output_saved_model_dir=output_saved_model_dir)
    print('Complete')