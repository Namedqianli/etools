# -*- coding:utf-8 -*-
import argparse
import os
import cv2
import numpy as np
import time

import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfmodel', type=str, required=True, help='path to tflite model')
    parser.add_argument('--input_type', type=str, default='float32', help='model input type')
    parser.add_argument('--output_type', type=str, default='float32', help='model output type')
    parser.add_argument('--input_shape', type=list, help='input shape')
    parser.add_argument('--input_layout', type=str, default='NCHW', help='input layout eg. NHWC or NCHW')
    parser.add_argument('--output_layout', type=str, default='NCHW', help='output layout eg. NHWC or NCHW')
    args = parser.parse_args()

    # read tf model
    interpreter = tf.lite.Interpreter(model_path=args.tfmodel)
    interpreter.allocate_tensors()
    
    output = interpreter.get_output_details()[0]  # Model has single output.
    print("model output info", output)
    input = interpreter.get_input_details()[0]  # Model has single input.
    print("model input info", input)
    input_data = tf.constant(1., shape=(1,32,96,1), dtype=tf.float32)
    print("model input shape", input_data.shape)
    interpreter.set_tensor(input['index'], input_data)
    interpreter.invoke()
    print(interpreter.get_tensor(output['index']).shape)

