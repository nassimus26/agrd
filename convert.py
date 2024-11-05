import sys
import tf2onnx
import onnx
import tensorflow as tf

if __name__ == '__main__':
    #model = tf.keras.models.load_model('fights.keras')
    import tf2onnx
    import onnx

    #input_signature = [tf.TensorSpec(v_input_shape, tf.float32, name='x')]
    # Use from_function for tf functions
    #model.output_names = ['output']

    onnx_model, _ = tf2onnx.convert.from_tflite('fights.tflite', opset=18)
    onnx.save(onnx_model, "fights.onnx")
    print('fights.onnx saved')
