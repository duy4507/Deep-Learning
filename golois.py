import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from tensorflow.keras import layers 
from tensorflow.keras import regularizers
import gc

from keras.callbacks import LearningRateScheduler
import math
import golois

from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Dense, Concatenate, Add, ReLU, BatchNormalization,AvgPool2D, MaxPool2D, GlobalAveragePooling2D, Reshape, Permute, Lambda, Flatten, Activation
from keras import backend as K
# To set learning rate


planes = 31
moves = 361
N = 10000
epochs = 1000
batch = 32
filters = 49
trunk = 28
blocks = 15

input_data = np.random.randint(2, size=(N, 19, 19, planes))
input_data = input_data.astype ('float32')

policy = np.random.randint(moves, size=(N,))
policy = keras.utils.to_categorical (policy)

value = np.random.randint(2, size=(N,))
value = value.astype ('float32')

end = np.random.randint(2, size=(N, 19, 19, 2))
end = end.astype ('float32')

groups = np.zeros((N, 19, 19, 1))
groups = groups.astype ('float32')

print ("getValidation", flush = True)
golois.getValidation (input_data, policy, value, end)

def bottleneck_block(x, expand=filters, squeeze=trunk):
  m = layers.Conv2D(expand, (1,1),kernel_regularizer=regularizers.l2(0.0001),use_bias = False)(x)
  m = layers.BatchNormalization()(m)
  m = layers.Activation('swish')(m)
  m1 = layers.DepthwiseConv2D((3,3), padding='same',kernel_regularizer=regularizers.l2(0.0001),use_bias = False)(m)
  m2 = layers.DepthwiseConv2D((5,5), padding='same',kernel_regularizer=regularizers.l2(0.0001),use_bias = False)(m)
  m = layers.Concatenate()([m1,m2])
  m = layers.BatchNormalization()(m)
  m = layers.Activation('swish')(m)
  m = layers.Conv2D(squeeze, (1,1),kernel_regularizer=regularizers.l2(0.0001),use_bias = False)(m)
  m = layers.BatchNormalization()(m)
  return layers.Add()([m, x])
def getModelMobileNet ():
  input = keras.Input(shape=(19, 19, 31), name='board')
  x = layers.Conv2D(trunk, 1, padding='same',kernel_regularizer=regularizers.l2(0.0001))(input)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)
  for i in range (blocks):
    x = bottleneck_block (x, filters, trunk)
  policy_head = layers.Conv2D(1, 1, activation='swish', padding='same',use_bias = False,kernel_regularizer=regularizers.l2(0.0001))(x)

  policy_head = layers.Flatten()(policy_head)
  policy_head = layers.Activation('softmax', name='policy')(policy_head)
  value_head = layers.GlobalAveragePooling2D()(x)
  value_head = layers.Dense(50, activation='swish',kernel_regularizer=regularizers.l2(0.0001))(value_head)
  value_head = layers.Dense(1, activation='sigmoid', name='value',
  kernel_regularizer=regularizers.l2(0.0001))(value_head)
  model = keras.Model(inputs=input, outputs=[policy_head, value_head])
  return model



eta_min=1e-8
eta_max=1e-3
T_max=epochs
lr = 0.001

model = getModelMobileNet()
model.summary ()

#tf.keras.utils.plot_model(model, 'KangGo.png', show_shapes=True)

model.compile(loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy' : 1.0, 'value' : 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'},
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

print ("getValidation", flush = True)
golois.getValidation (input_data, policy, value, end)

val_list=[]
for i in range (1, epochs + 1):
    print ('epoch ' + str (i)+' with lr = '+str(lr))
    golois.getBatch (input_data, policy, value, end, groups, i * N)

    history = model.fit(input_data,
                        {'policy': policy, 'value': value}, 
                        epochs=1, batch_size=batch,verbose=1)
    lr = eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * i / T_max)) / 2
    K.set_value(model.optimizer.lr, lr)

    if (i % 1 == 0):
        gc.collect ()
    if (i % 5 == 0):
        golois.getValidation (input_data, policy, value, end)
        val = model.evaluate (input_data,
                            [policy, value], verbose = 1, batch_size=batch)
        val_list.append(val)
        print ("val =", val)
        model.save ('test.h5')
        print('Model Saved')
val_df = pd.DataFrame(val_list,columns=['loss','policy_loss', 'value_loss', 'policy_categorical_accuracy', 'value_mse'])
val_df.to_csv('./val.csv')
