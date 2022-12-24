import numpy as np
import tensorflow_federated as tff
#emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

def restore_data_all(tff_dataset,num_clients):
  dataset = [] 

  total_data_size = 0
  ####################
  cnt = 0
  for id in range (num_clients):
      client_data = tff_dataset.create_tf_dataset_for_client(
        tff_dataset.client_ids[id])
    
      for i in iter(client_data):
        cnt = cnt+1
  total_size = cnt
  x = np.ones((total_size,32,32))  
  y = np.zeros((total_size))
  cnt = 0
  for id in range (num_clients):
    client_data = tff_dataset.create_tf_dataset_for_client(
      tff_dataset.client_ids[id])
    
    for i in iter(client_data):
      x[cnt,2:30,2:30] = i['pixels'].numpy()
      y[cnt] = i['label'].numpy()
      cnt = cnt +1
  dataset.append([x,y])
  return dataset

def restore_data_all_exclude(tff_dataset,num_clients):
  dataset = [] 

  total_data_size = 0
  ####################
  
  for idd in range (num_clients):
    cnt = 0
    for id in range (num_clients):
      if idd != id:  
        client_data = tff_dataset.create_tf_dataset_for_client(
          tff_dataset.client_ids[id])
      
        for i in iter(client_data):
          cnt = cnt+1
    total_size = cnt
    x = np.ones((total_size,32,32))  
    y = np.zeros((total_size))
    cnt = 0
    for id in range (num_clients):
      if idd != id:
        client_data = tff_dataset.create_tf_dataset_for_client(
          tff_dataset.client_ids[id])
        
        for i in iter(client_data):
          x[cnt,2:30,2:30] = i['pixels'].numpy()
          y[cnt] = i['label'].numpy()
          cnt = cnt +1
    dataset.append([x,y])
  return dataset

import random
def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects

def create_extreme_hetero_data(dataset_train,dataset_test,No_clients,n_class,n_train,n_test):
  dataset_train_par = init_list_of_objects(No_clients)
  dataset_test_par = init_list_of_objects(No_clients)
  for label in range(10):
    for sample in range(dataset_train[0][0].shape[0]):
      if dataset_train[0][1][sample] ==label:
        dataset_train_par[label].append(dataset_train[0][0][sample])

  for label in range(10):
    for sample in range(dataset_test[0][0].shape[0]):
      if dataset_test[0][1][sample] ==label:
        dataset_test_par[label].append(dataset_test[0][0][sample])

  for label in range(10):
    dataset_train_par[label] = np.asarray(dataset_train_par[label])

  for label in range(10):
    dataset_test_par[label] = np.asarray(dataset_test_par[label])

  data_train_new = init_list_of_objects(No_clients)
  label_train_new = init_list_of_objects(No_clients)

  data_test_new = init_list_of_objects(No_clients)
  label_test_new = init_list_of_objects(No_clients)
  for client in range(No_clients):
    
    xx = range(10)
    xx = sorted(xx, key = lambda x: random.random() )
    label_indx = xx[0:n_class]
    cnt = 0
    for indx in label_indx:
      if cnt ==0:
        data_train_new[client] = dataset_train_par[indx][np.random.randint(0,dataset_train_par[indx].shape[0],n_train)]
        label_train_new[client] = indx*np.ones((n_train))

        data_test_new[client] = dataset_test_par[indx][np.random.randint(0,dataset_test_par[indx].shape[0],n_test)]
        label_test_new[client] = indx*np.ones((n_test))
      else:
        data_train_new[client] = np.concatenate([data_train_new[client],dataset_train_par[indx][np.random.randint(0,dataset_train_par[indx].shape[0],n_train)]])
        label_train_new[client] = np.concatenate([label_train_new[client], indx*np.ones((n_train))])

        data_test_new[client] = np.concatenate([data_test_new[client],dataset_test_par[indx][np.random.randint(0,dataset_test_par[indx].shape[0],n_test)]])
        label_test_new[client] = np.concatenate([label_test_new[client], indx*np.ones((n_test))])
      cnt +=1
  return data_train_new, label_train_new, data_test_new, label_test_new