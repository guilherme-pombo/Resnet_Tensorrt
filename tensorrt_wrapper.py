import uff
import os
import tensorflow as tf
import pycuda.driver as cuda
import keras.backend as K
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.applications import resnet50
import numpy as np
import tensorrt as trt
from tensorrt.parsers import uffparser


# your model should already have been trained, and you'd be using it for inference only
# set learning phase to Testing (0) so that all untrainable nodes are excluded from the UFF model
K.set_learning_phase(0)

# very important -- gotta set the right CUDA architecture otherwise you can't build the engine
# V100 -> 70, don't bother using this in K80s or 1080
os.environ["CUDA_ARCH"] = "70"
# how much memory to allocate for the engine -- 1GB -- can go a bit higher on V100
MAX_WORKSPACE_SIZE = 1 << 30
# maximum batch size allowed -- 128 was the best
MAX_BATCH_SIZE = 128
# what datatype to represent the matrices on the GPU
TRT_DATATYPE = trt.infer.DataType.FLOAT
trt_logger = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)


class TensorrtWrapper:

    def __init__(self, uff_path, model_input_name, model_output_name, batch_size=MAX_BATCH_SIZE):
        self.batch_size = batch_size
        self.stream = cuda.Stream()
        self.model_input_name = model_input_name
        self.model_output_name = model_output_name

        print("Creating tensorrt context....")
        self.context = self.parse_uff_model(uff_path=uff_path)
        self.output = self.d_input = self.d_output = self.bindings = None
        print("Allocating memory arrays....")
        self.allocate_memory_arrays()

    def parse_uff_model(self, uff_model=None, uff_path=None):
        assert uff_model or uff_path, "Must pass in either a UFF model or the path to an UFF model in disk"
        if uff_path:
            with open(uff_path, 'rb') as uff_file:
                uff_model = uff_file.read()
        parser = uffparser.create_uff_parser()
        # input_1
        parser.register_input(self.model_input_name, (3, 224, 224), 0)
        # dense_2/Sigmoid
        parser.register_output(self.model_output_name)
        engine = trt.utils.uff_to_trt_engine(logger=trt_logger,
                                             stream=uff_model,
                                             parser=parser,
                                             max_batch_size=MAX_BATCH_SIZE,
                                             max_workspace_size=MAX_WORKSPACE_SIZE,
                                             datatype=TRT_DATATYPE)
        context = engine.create_execution_context()
        return context

    def allocate_memory_arrays(self):
        # load engine
        engine = self.context.get_engine()
        assert (engine.get_nb_bindings() == 2), "Wrong engine configuration for our task, please check tensorrt" \
                                                " documentation before using this"
        # create output array to receive data
        dims = engine.get_binding_dimensions(1).to_DimsCHW()
        elt_count = dims.C() * dims.H() * dims.W() * self.batch_size
        # create a sample batch image to define how much memory we need to allocate
        input_img = np.random.rand(self.batch_size, 224, 224, 3).astype(np.float32)
        # Allocate pagelocked memory
        self.output = cuda.pagelocked_empty(elt_count, dtype=np.float32)
        print("Image size: {}".format(input_img.size))
        # alocate device memory
        self.d_input = cuda.mem_alloc(self.batch_size * input_img.size * input_img.dtype.itemsize)
        self.d_output = cuda.mem_alloc(self.batch_size * self.output.size * self.output.dtype.itemsize)
        self.bindings = [int(self.d_input), int(self.d_output)]

    def run_prediction(self, input_img):
        """
        Use this to run the actual inference on the device
        :return:
        """
        # transfer input data to device
        cuda.memcpy_htod_async(self.d_input, input_img, self.stream)
        # execute model
        self.context.enqueue(self.batch_size, self.bindings, self.stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)


###################################
### UTILS #########################
###################################


def convert_keras_to_uff_model(model, uff_model_path):
    # have to make BatchNorm layers untrainable since they're not yet supported by tensorrt
    for entry in model.layers:
        if 'bn' in entry:
            entry.trainable = False

    model_input_name = model.input.name.strip(':0')
    model_output_name = model.output.name.strip(':0')
    input_size = model.input.shape
    print(input_size)
    graph = tf.get_default_graph().as_graph_def()
    init = tf.global_variables_initializer()
    sess = K.get_session()
    sess.run(init)

    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graph, [model_output_name])
    frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
    uff_model = uff.from_tensorflow(frozen_graph, [model_output_name])
    with open(uff_model_path, 'wb') as dump:
        dump.write(uff_model)

    return model_input_name, model_output_name


def process_img_example():
    test_image = image.load_img('example_image.jpg', target_size=(224, 224, 3))
    test_image = image.img_to_array(test_image)
    processed_im = preprocess_input(np.expand_dims(test_image, 0))[0, :, :, :]
    processed_im = np.transpose(processed_im, axes=(2, 0, 1))
    # gotta make the image matrix contiguous
    processed_im = processed_im.copy(order='C')
    return processed_im


if __name__ == '__main__':
    # small example

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
