import tensorflow as tf
import numpy as np
import graph_tools as Gr
from net import CNN

def local_update(E=3, model=CNN(),x_data=None,y_data=None,batch_size=32):

  for epoch in range(E):      
          info = model.fit(x_data, y_data ,batch_size=batch_size,
      epochs=1,
      verbose=0,
      callbacks=None,
      validation_split=0.0,
      validation_data=None,
      shuffle=True,)

  client_var_FedAvg = model.get_weights()
  return client_var_FedAvg

def server_update(method='FedAvg',delta_sum=None,N_active_clients=None, client_update=None, global_update=None,Graph=None,Gradient_list=None,mu=None,):

  """

The aggregation process. mu is only valid when method is G-Fedfilt.
  """

  if method =='FedAvg':

    Delta = []
    for (item1, item2) in zip(client_update, global_update):
        Delta.append( item2 - item1)


    sum_list =[]
    if delta_sum is None:
      # weight = data_size / total_data_size
      weight = 1/N_active_clients
      delta_sum = [weight * x for x in Delta]
    else:
      for (item1, item2) in zip(Delta, delta_sum):
        sum_list.append( weight*item1 + item2)

      delta_sum = sum_list
    global_temp = []
    for (item1, item2) in zip(global_update, delta_sum):
          global_temp.append( item1 - item2)  

    global_update = global_temp
    return global_update
  ###################################################

  elif method=='G-Fedfilt':
     Grad = Gr.soft_filt(Gradient_list,Graph,Gr.h_s,mu,0) 
  ###################################################
     return Grad
  else:
    exit('Please provide a valid aggregation method (e.g.: \'FedAvg\',\'G-Fedfilt\')')