import tensorflow as tf
import tfcoreml
import numpy as np
import PIL
import requests
from io import BytesIO
from matplotlib.pyplot import imshow
from coremltools.proto import FeatureTypes_pb2 as _FeatureTypes_pb2
import coremltools

"""FIND FROZEN GRAPH INFO"""
tf_model_path = "./retrained_graph.pb"
with open(tf_model_path , 'rb') as f:
    serialized = f.read()
tf.reset_default_graph()
original_gdef = tf.GraphDef()
original_gdef.ParseFromString(serialized)

with tf.Graph().as_default() as g:
    tf.import_graph_def(original_gdef, name ='')
    ops = g.get_operations()
    N = len(ops)
    for i in [0,1,2,N-3,N-2,N-1]:
        print('\n\nop id {} : op type: "{}"'.format(str(i), ops[i].type))
        print('input(s):')
        for x in ops[i].inputs:
            print("name = {}, shape: {}, ".format(x.name, x.get_shape()))
        print('\noutput(s):'),
        for x in ops[i].outputs:
            print("name = {}, shape: {},".format(x.name, x.get_shape()))

""" CONVERT TF TO CORE ML """
# # Supply a dictionary of input tensors' name and shape (with batch axis)
input_tensor_shapes = {"input:0":[1,224,224,3]} # batch size is 1
#providing the image_input_names argument converts the input into an image for CoreML
image_input_name = ['input:0']
# Output CoreML model path
coreml_model_file = './inception_v1.mlmodel'
# The TF model's ouput tensor name
output_tensor_names = ['final_result:0']
# class label file: providing this will make a "Classifier" CoreML model
class_labels = 'retrained_labels.txt'

#Call the converter. This may take a while
coreml_model = tfcoreml.convert(
        tf_model_path=tf_model_path,
        mlmodel_path=coreml_model_file,
        input_name_shape_dict=input_tensor_shapes,
        output_feature_names=output_tensor_names,
        image_input_names = image_input_name,
        class_labels = class_labels)

img_url = 'http://im.uniqlo.com/images/jp/sp/goods/307068/item/62_307068_mb.jpg'
response = requests.get(img_url)
img = PIL.Image.open(BytesIO(response.content))
imshow(np.asarray(img))


img = img.resize([224, 224], PIL.Image.ANTIALIAS)
coreml_inputs = {'input__0': img}
coreml_output =  coreml_model.predict(coreml_inputs, useCPUOnly=True)
coreml_pred_dict = coreml_output['final_result__0']
coreml_predicted_class_label = coreml_output['classLabel']

# for getting TF prediction we get the numpy array of the image
img_np = np.array(img).astype(np.float32)
print('image shape:', img_np.shape)
print('first few values: ', img_np.flatten()[0:4], 'max value: ', np.amax(img_np))
img_tf = np.expand_dims(img_np, axis=0)  # now shape is [1,224,224,3] as required by TF

# Evaluate TF and get the highest label
tf_input_name = 'input:0'
tf_output_name = 'final_result:0'
with tf.Session(graph=g) as sess:
    tf_out = sess.run(tf_output_name,
                      feed_dict={tf_input_name: img_tf})
tf_out = tf_out.flatten()
idx = np.argmax(tf_out)
label_file = class_labels
with open(label_file) as f:
    labels = f.readlines()

# print predictions
print('\n')
print("CoreML prediction class = {}, probabiltiy = {}".format(coreml_predicted_class_label,
                                                              str(coreml_pred_dict[coreml_predicted_class_label])))
print("TF prediction class = {}, probability = {}".format(labels[idx],
                                                          str(tf_out[idx])))



img_tf = (2.0/255.0) * img_tf - 1
with tf.Session(graph = g) as sess:
    tf_out = sess.run(tf_output_name,
                      feed_dict={tf_input_name: img_tf})
tf_out = tf_out.flatten()
idx = np.argmax(tf_out)


print("TF prediction class = {}, probability = {}".format(labels[idx],
                                            str(tf_out[idx])))
print("CoreML prediction class = {}, probabiltiy = {}".format(coreml_predicted_class_label,
                                                              str(coreml_pred_dict[coreml_predicted_class_label])))

spec = coremltools.models.utils.load_spec(coreml_model_file)
if spec.WhichOneof('Type') == 'neuralNetworkClassifier':
  nn = spec.neuralNetworkClassifier
if spec.WhichOneof('Type') == 'neuralNetwork':
  nn = spec.neuralNetwork
if spec.WhichOneof('Type') == 'neuralNetworkRegressor':
  nn = spec.neuralNetworkRegressor

preprocessing = nn.preprocessing[0].scaler
print('channel scale: ', preprocessing.channelScale)
print('blue bias: ', preprocessing.blueBias)
print('green bias: ', preprocessing.greenBias)
print('red bias: ', preprocessing.redBias)

inp = spec.description.input[0]
if inp.type.WhichOneof('Type') == 'imageType':
  colorspace = _FeatureTypes_pb2.ImageFeatureType.ColorSpace.Name(inp.type.imageType.colorSpace)
  print('colorspace: ', colorspace)

coreml_model = tfcoreml.convert(
        tf_model_path=tf_model_path,
        mlmodel_path=coreml_model_file,
        input_name_shape_dict=input_tensor_shapes,
        output_feature_names=output_tensor_names,
        image_input_names = image_input_name,
        class_labels = class_labels,
        red_bias = -1,
        green_bias = -1,
        blue_bias = -1,
        image_scale = 2.0/255.0)

coreml_output = coreml_model.predict(coreml_inputs, useCPUOnly=True)
coreml_pred_dict = coreml_output['final_result__0']
coreml_predicted_class_label = coreml_output['classLabel']
print("CoreML prediction class = {}, probability = {}".format(coreml_predicted_class_label,
                                            str(coreml_pred_dict[coreml_predicted_class_label])))