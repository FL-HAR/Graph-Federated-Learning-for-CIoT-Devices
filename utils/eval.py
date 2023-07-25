import tensorflow as tf
import numpy as np
def evaluate_personalization(ww,data_test,label_test,keras_model,client_id):

  keras_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  
  )
  keras_model.set_weights(ww)
  client_xdata = np.expand_dims(data_test[client_id],-1)
  client_ydata = label_test[client_id]
  return keras_model.evaluate(client_xdata,client_ydata)

def evaluate_generalization(client_weights,dataset_test,keras_model):
  
  keras_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  
  )
  keras_model.set_weights(client_weights)
  client_xdata = np.expand_dims(dataset_test[0],axis=-1)
  client_ydata = dataset_test[1]
  return keras_model.evaluate(client_xdata,client_ydata)
