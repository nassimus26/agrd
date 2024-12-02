# -*- coding: utf-8 -*-
"""genral_training_example.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DCSO2q3TZCOSVLkz_EOyxcGD9B7zDjug
"""
# from rknn.api import RKNN

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import save
from numpy import load
from sklearn.model_selection import train_test_split
from color import PrintColored
layers = tf.keras.layers
models = tf.keras.models
losses = tf.keras.losses
optimizers = tf.keras.optimizers
metrics = tf.keras.metrics
utils = tf.keras.utils
callbacks = tf.keras.callbacks
layers = tf.keras.layers
models = tf.keras.models
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
losses = tf.keras.losses
optimizers = tf.keras.optimizers
metrics = tf.keras.metrics
utils = tf.keras.utils
callbacks = tf.keras.callbacks
plot_model = tf.keras.utils.plot_model

def cutSave(main_dir, mod):
    i = 0
    fights_train_dest = main_dir+'_'+mod+'_fights_train'+'.npy'
    labels_train_dest = main_dir+'_'+mod+'_labels_train'+'.npy'
    if mod == 'train':
        if (False & os.path.exists(fights_train_dest) &
            os.path.exists(labels_train_dest)):
            print('loading train arrays')
            fights_train_ = load(fights_train_dest)
            labels_train_ = load(labels_train_dest)
            fights_train[:] = fights_train_
            labels_train[:] = labels_train_
        else:
            print('retrieving train arrays')
            for x in os.listdir(main_dir):
                td = main_dir+x+'/'
                if os.path.isdir(td):
                    #for y in os.listdir(main_dir+x+'/'):
                        #print(y)
                    for file in os.listdir(td):
                        #print('img>>> ' + file)
                        fl = os.path.join(td, file)
                        videos = capture(fl)
                        if mod == 'train':
                            fights_train[i][:][:] = videos
                            if x =='fight':
                                labels_train[i]=1
                            else:
                                labels_train[i]=0
                        elif mod =='test':
                            fights_test[i][:][:] = videos
                            if x =='fight':
                                labels_test[i]=1
                            else:
                                labels_test[i]=0
                        elif mod =='val':
                            fights_val[i][:][:] = videos
                            if x =='fight':
                                labels_vals[i]=1
                            else:
                                labels_vals[i]=0
                        i +=1
                        if ((i==dataset_size/2) & (x == 'fight')) or i==dataset_size:
                            break
                    if i==dataset_size:
                        break
            if mod == 'train':
                print('saving train arrays')
#                save(mod+'_fights_train'+'.npy', fights_train)
#                save(mod+'_labels_train'+'.npy', labels_train)

nbr_frame = 6
img_width = 224
img_height = 224
img_size = (img_height, img_width)
input_shape = img_size + (3,)
full_input_shape = (nbr_frame,) + input_shape
print(full_input_shape)

dataset_size = 100
fights_train = np.zeros((dataset_size,) + full_input_shape, dtype=np.float32)
fights_val = fights_train
labels_train = np.empty(dataset_size, dtype=np.int_)
labels_vals = np.empty(dataset_size, dtype=np.int_)
def capture(filename):
    frames = np.zeros( (20,) +input_shape, dtype=np.float32)
    i = 0
    vc = cv2.VideoCapture(filename)
    while i < 20:
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            print('unreadable '+filename)
        try:
            frame.shape
        except:
            print(' ' + str(i) + ' ' + filename + ' ')
        frm = cv2.resize(frame, (img_width, img_height))
        frm = np.expand_dims(frm, axis=0)
        #if np.max(frm)>1:
        #    frm = frm/255.0
        # print(frm.shape)
        #frm = frm[0].transpose(2,0,1)
        frames[i][:] = frm
      # print(frames[i])
        i +=1
        if i==(nbr_frame*8):
            break
   # print(i)
    step = int(i/nbr_frame)
    resf = np.empty( (nbr_frame,)+input_shape )
    k = 0
   # print(step)
    while k<nbr_frame:
        j = int(step*k)
        resf[k][:] = frames[j]
        k = k+1
 #  print(resf)
    return resf
cutSave('./trainm/',"train")

#cut_save('./testm/',"test")
#plt.imshow(fights_test[19][5])
#plt.show()

class AccuracyHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

history = AccuracyHistory()
earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=8,min_delta=1e-5, verbose=0, mode='min')
mcp_save = callbacks.ModelCheckpoint('fights.keras', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='val_loss',patience=10, verbose=2, factor=0.5,min_lr=0.0000001)

np.random.seed(1234)
num_classes = 2
tf.keras.backend.set_image_data_format(
    'channels_last'
)

v_input_shape = (0,) + full_input_shape
#model.save("fights.keras")

#model.save('ss')

import tensorflow as tf

#model = tf.keras.models.load_model('fights.keras')


#model = tf.keras.models.load_model('fights.keras')
interpreter = tf.lite.Interpreter(model_path='fights.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('importing rknn')
from rknn.api import RKNN
print('rknn imported')
TEST_CASES = dataset_size
rknn = RKNN(verbose=True)
#rknn.load_rknn('fights.rknn')
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
rknn.config(
        mean_values=mean,
        std_values=std,
        target_platform='rv1106',
       # dynamic_input=[[[1, 3, img_height, img_width],
                       #[1, img_height, img_width, 3],
                       #[1, img_height, img_width, 3],
                       #[1, img_height, img_width, 3],
                       #[1, img_height, img_width, 3],
                       #[1, img_height, img_width, 3],
                       #[1, img_height, img_width, 3],
                       #[1, img_height, img_width, 3]
        #                ]]
    )
#ret = rknn.load_tflite(model='fights.tflite', input_is_nchw=False)
input = [1, 3, 300, 300]
#inputs = np.repeat( [input], 6, axis=0)
print('loading model for conversion')
ret = rknn.load_onnx(model='fights.onnx', input_size_list=[
[1, 3, 420, 300],
[1, 3, 420, 300],
[1, 3, 420, 300],
[1, 3, 420, 300],
[1, 3, 420, 300],
[1, 3, 420, 300],
[1, 3, 420, 300],
[1, 3, 420, 300],
])
print('loaded')
if ret != 0:
    print('Load model failed!')
    exit(ret)
print('done')
print('############--> Building model')
ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
if ret != 0:
    print('Build model failed!')
    exit(ret)
ret = rknn.export_rknn('fights.rknn')
print('############ done')

ret = rknn.init_runtime()
if ret != 0:
    print('Init runtime environment failed!')
    exit(ret)
BigError = 0
SmallError = 0
for i in range(TEST_CASES):
  inputs = [np.expand_dims(_fights_train[0][i], 0),
            #               np.expand_dims(_fights_train[1][i], 0),
            #               np.expand_dims(_fights_train[2][i], 0),
            #               np.expand_dims(_fights_train[3][i], 0),
            #                np.expand_dims(_fights_train[4][i], 0),
            #    np.expand_dims(_fights_train[5][i], 0),
                        #    np.expand_dims(_fights_train[6][i], 0),
                        #    np.expand_dims(_fights_train[7][i], 0),
            ]
  #for v in range(8):
   #   inputs[v] = inputs[v].transpose(0, 3, 1, 2)
      #print(inputs[v].shape)
  #print('input_details')
  #print(input_details[0])
  from torchvision import datasets, transforms

  outputs = rknn.inference(inputs=[inputs[0],
                                   inputs[1],
                                   inputs[2],
                                   inputs[3],
                                   inputs[4],
                                   inputs[5],
                         #          inputs[6],
                        #           inputs[7],
                                        ])
 # expected = model.predict(inputs)
  for j in range(6):
    interpreter.set_tensor(input_details[j]['index'], np.expand_dims(_fights_train[j][i], 0))
  interpreter.invoke()
  result = interpreter.get_tensor(output_details[0]["index"])
 # print(expected)
  print('TF [' + str(outputs[0][0][0]) + ',' + str(outputs[0][0][1]) + ']')
  print(result)
  expected = outputs[0][0][0] - outputs[0][0][1]
  res = result[0][0]-result[0][1]
  print = PrintColored()
  color = print.GREEN
  if expected*res<0:
      color = print.RED
      BigError +=1
  else:
      if abs(expected)-abs(res)>0.23:
          color = print.YELLOW_DARK
          SmallError += 1
  print(str(expected) + ' ' + str(res), color=color)
  print("", default_color=print.DEFAULT)

  #np.testing.assert_almost_equal(expected, result, decimal=5)
  #print("Done. The result of TensorFlow matches the result of TensorFlow Lite.")
  # Please note: TfLite fused Lstm kernel is stateful, so we need to reset
  # the states.
  # Clean up internal states.
  interpreter.reset_all_variables()

print('Nbr Big Errors = '+str(BigError)+
      ' Small Error = ' + str(SmallError), color=print.BLUE)
print("", default_color=print.DEFAULT)

model = tf.keras.models.load_model('fights.keras')
import tf2onnx
import onnx

v_input_shape = (0,) + full_input_shape
input_signature = [tf.TensorSpec(v_input_shape, tf.float32, name='x')]
# Use from_function for tf functions
model.output_names=['output']
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=18)
#onnx.save(onnx_model, "fights.onnx")
print('fights.onnx saved')
exit(0)
means = [0,0,0]
#means = np.repeat(means[np.newaxis,...], 180, axis=0)
#means = np.repeat(means[np.newaxis,...], 150, axis=0)

#rknn.config( target_platform='rv1106', mean_values=[means], std_values=[means], data_format='')
#ret = rknn.load_onnx(model='fights.onnx', inputs=['x'], input_size_list=[[10, 150, 180, 3]])


acc = history.acc
val_acc = history.val_acc
loss = history.loss
val_loss = history.val_loss
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

score = model.evaluate(fights_test, y_test, batch_size=3)
score

from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

Y_pred = model.predict(fights_test , batch_size=1)

yprd = Y_pred > 0.5
yprd

ypredicted = []
for zero,one in yprd:
    if zero == True:
        ypredicted.append(0)
    else:
        ypredicted.append(1)

ypredicted

y_test

y = []

for zero,one in y_test:
    if zero == True:
        y.append(0)
    else:
        y.append(1)

confusion = confusion_matrix(y,ypredicted)
confusion.shape

print_confusion_matrix(confusion, [0,1], figsize = (30,15), fontsize=16)

print('Classification Report')
print(classification_report(y, ypredicted, target_names=['no-violance','violance']))