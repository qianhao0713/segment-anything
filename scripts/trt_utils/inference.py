import os
import time

import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import numpy as np
import pynvml

from . import common as common
from . import engine as engine_utils # TRT Engine creation/save/load utils
from . import model as model_utils # UFF conversion uttils

# TensorRT logger singleton
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

class TRTInference(object):
    """Manages TensorRT objects for model inference."""
    def __init__(self, trt_engine_path, trt_engine_datatype=trt.DataType.FLOAT, batch_size=1, dynamic_shape={}, dynamic_shape_value={}, is_torch_infer=False, device=None):
        """Initializes TensorRT objects needed for model inference.
        Args:
            trt_engine_path (str): path where TensorRT engine should be stored
            trt_engine_datatype (trt.DataType):
                requested precision of TensorRT engine used for inference
            batch_size (int): batch size for which engine
                should be optimized for
        """

        # We first load all custom plugins shipped with TensorRT,
        # some of them will be needed during inference
        import torch
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')

        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        # TRT engine placeholder
        self.trt_engine = None
        # Display requested engine settings to stdout
        print("TensorRT inference engine settings:")
        print("  * Inference precision - {}".format(trt_engine_datatype))
        print("  * Max batch size - {}\n".format(batch_size))

        # If engine is not cached, we need to build it
        if not os.path.exists(trt_engine_path):
            raise Exception('tensorRT engine file not exist')
        self.trt_engine = engine_utils.load_engine(
            self.trt_runtime, trt_engine_path)
        # Execution context is needed for inference

        self.context = self.trt_engine.create_execution_context()

        # This allocates memory for network inputs/outputs on both CPU and GPU
        if is_torch_infer:
            self.inputs, self.outputs, self.bindings, self.stream = \
                common.allocate_buffers_torch(self.trt_engine, self.context, dynamic_shape, dynamic_shape_value, device)
        else:
            self.inputs, self.outputs, self.bindings, self.stream = \
                common.allocate_buffers(self.trt_engine, self.context, dynamic_shape, dynamic_shape_value)
        # Allocate memory for multiple usage [e.g. multiple batch inference]
        input_volume = trt.volume(model_utils.ModelData.INPUT_SHAPE)
        self.numpy_array = np.zeros((self.trt_engine.max_batch_size, input_volume))
        
    def torch_inference(self, dict_input):
        for i, binding in enumerate(self.trt_engine):
            if self.trt_engine.binding_is_input(binding):
                self.inputs[i][...] = dict_input[binding]
        common.do_inference_torch(self.context, bindings=self.bindings)
        return self.outputs

    def inference(self, dict_input, h2d=True, d2h=True, device_input = [], device_input_size = []):
        """
        Do inference by tensorrt builded engine.

        Parameters
        ----------
        test_data : numpy tensor
            Model input tensor
        """
        # Numpy dtype should be float32
        # assert test_data.dtype == np.float32
        # self.inputs[0].host = test_data
        device_index_input = {}
        for i, binding in enumerate(self.trt_engine):
            if self.trt_engine.binding_is_input(binding):
                if binding in device_input:
                    device_index_input[i] = (dict_input[binding], device_input_size[i])
                else:
                    self.inputs[i].host = dict_input[binding]
        output = common.do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream, h2d=h2d, d2h=d2h, device_input=device_index_input)
        return output

    def infer(self, image_path):
        """Infers model on given image.
        Args:
            image_path (str): image to run object detection model on
        """

        # Load image into CPU
        img = self._load_img(image_path)

        # Copy it into appropriate place into memory
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, img.ravel())

        # When infering on single image, we measure inference
        # time to output it to the user
        inference_start_time = time.time()

        # Fetch output from the model
        [detection_out, keepCount_out] = common.do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream)

        # Output inference time
        print("TensorRT inference time: {} ms".format(
            int(round((time.time() - inference_start_time) * 1000))))

        # And return results
        return detection_out, keepCount_out

    def infer_webcam(self, arr):
        """Infers model on given image.
        Args:
            arr (numpy array): image to run object detection model on
        """

        # Load image into CPU
        img = self._load_img_webcam(arr)

        # Copy it into appropriate place into memory
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, img.ravel())

        # When infering on single image, we measure inference
        # time to output it to the user
        inference_start_time = time.time()

        # Fetch output from the model
        [detection_out, keepCount_out] = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream)

        # Output inference time
        print("TensorRT inference time: {} ms".format(
            int(round((time.time() - inference_start_time) * 1000))))

        # And return results
        return detection_out, keepCount_out

    def infer_batch(self, image_paths):
        """Infers model on batch of same sized images resized to fit the model.
        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """

        # Verify if the supplied batch size is not too big
        max_batch_size = self.trt_engine.max_batch_size
        actual_batch_size = len(image_paths)
        if actual_batch_size > max_batch_size:
            raise ValueError(
                "image_paths list bigger ({}) than engine max batch size ({})".format(actual_batch_size, max_batch_size))

        # Load all images to CPU...
        imgs = self._load_imgs(image_paths)
        # ...copy them into appropriate place into memory...
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, imgs.ravel())

        # ...fetch model outputs...
        [detection_out, keep_count_out] = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=max_batch_size)
        # ...and return results.
        return detection_out, keep_count_out

    def _load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image).reshape(
            (im_height, im_width, model_utils.ModelData.get_input_channels())
        ).astype(np.uint8)

    def _load_imgs(self, image_paths):
        batch_size = self.trt_engine.max_batch_size
        for idx, image_path in enumerate(image_paths):
            img_np = self._load_img(image_path)
            self.numpy_array[idx] = img_np
        return self.numpy_array

    def _load_img_webcam(self, arr):
        image = Image.fromarray(np.uint8(arr))
        model_input_width = model_utils.ModelData.get_input_width()
        model_input_height = model_utils.ModelData.get_input_height()
        # Note: Bilinear interpolation used by Pillow is a little bit
        # different than the one used by Tensorflow, so if network receives
        # an image that is not 300x300, the network output may differ
        # from the one output by Tensorflow
        image_resized = image.resize(
            size=(model_input_width, model_input_height),
            resample=Image.BILINEAR
        )
        img_np = self._load_image_into_numpy_array(image_resized)
        # HWC -> CHW
        img_np = img_np.transpose((2, 0, 1))
        # Normalize to [-1.0, 1.0] interval (expected by model)
        img_np = (2.0 / 255.0) * img_np - 1.0
        img_np = img_np.ravel()
        return img_np

    def _load_img(self, image_path):
        image = Image.open(image_path)
        model_input_width = model_utils.ModelData.get_input_width()
        model_input_height = model_utils.ModelData.get_input_height()
        # Note: Bilinear interpolation used by Pillow is a little bit
        # different than the one used by Tensorflow, so if network receives
        # an image that is not 300x300, the network output may differ
        # from the one output by Tensorflow
        image_resized = image.resize(
            size=(model_input_width, model_input_height),
            resample=Image.BILINEAR
        )
        img_np = self._load_image_into_numpy_array(image_resized)
        # HWC -> CHW
        img_np = img_np.transpose((2, 0, 1))
        # Normalize to [-1.0, 1.0] interval (expected by model)
        img_np = (2.0 / 255.0) * img_np - 1.0
        img_np = img_np.ravel()
        return img_np

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

