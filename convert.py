import os
import argparse
import json

import numpy as np
import torch
import onnx
import onnxruntime

try:
    import tensorflow as tf
except ImportError:
    print("Warning: Tensorflow not installed. This is required when exporting to tflite")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def print_final_message(model_path):
    success_msg = f"\033[92mModel converted at {model_path} \033[0m"
    try:
        import requests
        joke = json.loads(requests.request("GET", "https://api.chucknorris.io/jokes/random?category=dev").text)["value"]
        print(f"{success_msg}\n\nNow go read a Chuck Norris joke:\n\033[1m{joke}\033[0m")
    except ImportError:
        print(success_msg)


def convert_tf_saved_model(onnx_model, output_folder):
    from onnx_tf.backend import prepare
    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph(output_folder)  # export the model


def convert_tf_to_lite(model_dir, output_path):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)  # path to the SavedModel directory
    # This is needed for TF Select ops: Cast, RealDiv
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]

    tflite_model = converter.convert()

    # Save the model.
    with open(output_path, 'wb') as f:
        f.write(tflite_model)


def validate_tflite_output(model_path, input_data, output_array):
    interpreter = tf.lite.Interpreter(model_path=model_path)

    output = interpreter.get_output_details()[0]  # Model has single output.
    input = interpreter.get_input_details()[0]  # Model has single input.
    interpreter.resize_tensor_input(input['index'], input_data.shape)
    interpreter.allocate_tensors()

    input_data = tf.convert_to_tensor(input_data, np.float32)
    interpreter.set_tensor(input['index'], input_data)
    interpreter.invoke()

    out = interpreter.get_tensor(output['index'])
    np.testing.assert_allclose(out, output_array, rtol=1e-03, atol=1e-05)


def convert(checkpoint_path, export_tensorflow):
    output_folder = "converted_models"
    model = torch.load(checkpoint_path, map_location='cpu')
    model.eval()

    # Input to the model
    x = torch.randn(1, 10, 54, 2, requires_grad=True)
    numpy_x = to_numpy(x)

    torch_out = model(x)
    numpy_out = to_numpy(torch_out)

    model_path = f"{output_folder}/spoter.onnx"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Export the model
    torch.onnx.export(model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      model_path,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['output'],
                      dynamic_axes={'input': [1]}) # the model's output names

    # Validate conversion
    onnx_model = onnx.load(model_path)
    a = onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(model_path)

    # compute ONNX Runtime output prediction
    ort_inputs = { ort_session.get_inputs()[0].name: to_numpy(x) }
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(numpy_out, ort_outs[0], rtol=1e-03, atol=1e-05)

    if export_tensorflow:
        saved_model_dir = f"{output_folder}/tf_saved"
        tflite_model_path = f"{output_folder}/spoter.tflite"
        convert_tf_saved_model(onnx_model, saved_model_dir)
        convert_tf_to_lite(saved_model_dir, tflite_model_path)
        validate_tflite_output(tflite_model_path, numpy_x, numpy_out)

    print_final_message(output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', help='Checkpoint Path')
    parser.add_argument('-tf', '--export_tensorflow', help='Export Tensorflow apart from ONNX', action='store_true')
    args = parser.parse_args()
    convert(args.checkpoint_path, args.export_tensorflow)
