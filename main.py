import argparse
import tensorflow_federated as tff
import sys
import os
from IPython import display
import random
root = os.getcwd()
sys.path.append(root+'/utils')
sys.path.append(root+'/network')

from model_updates import local_update, server_update
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import graph_tools as Gr
from eval import evaluate_manual_data, evaluate
from net import CNN
from dataset import restore_data_all, create_extreme_hetero_data
def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset') ## the only current dataset is MNIST
    parser.add_argument('--model', type=str, default='CNN', help='model') ## the only current model is CNN
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for local training')
    parser.add_argument('--E', type=int, default=3, help='Local training rounds')
    parser.add_argument('--R', type=int, default=400, help='Total number of communication rounds in the framework')
    parser.add_argument('--N_clients', type=int, default=30, help='number of clients in the framework') ## for the paper graph this must be set to 20
    parser.add_argument('--paper_graph', type=str, default='N', help='whether to use the graph used in the original paper or a random sensor graph')
    parser.add_argument('--save_path', type=str, default='report', help='path to save results')
    parser.add_argument('--label_hetro', type=int, default=4, help='indicates label heterogeneity: 4 means each client only has 4 labels of the data')
    parser.add_argument('--Filter_sim', type=int, default=1, help='Number of filters to simulate')
    parser.add_argument('--mu_param', type=int, default=10, help='The tuning parameter of the graph filter')
    
    ##################################################
    parser.add_argument("-f", "--file", required=False) # this particular code is essential for the parser to work in google colab
    ##################################################
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if args.paper_graph=='Y':
      print(30*"#"+" Graph from the paper "+ 30*"#")
      N_clients = args.N_clients
      Gph = Gr.graph_in_the_paper()
      plt.pause(1)
      
    else:
      N_clients = args.N_clients
      Gph = Gr.graph_ex(N_clients)


    ## Loading data
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    dataset_train = restore_data_all(emnist_train,400)
    dataset_test = restore_data_all(emnist_test,400)
    data_train_new, label_train_new, data_test_new, label_test_new = create_extreme_hetero_data(dataset_train,dataset_test,N_clients,args.label_hetro,24,8)

    ## preparing the settings
    e_max = args.R
    accuracy_FedAvg_general = np.zeros((e_max))
    accuracy_FedAvg_local = np.zeros((e_max))
    accuracy_Fedfilt_general = [np.zeros((e_max)),np.zeros((e_max)),np.zeros((e_max)),np.zeros((e_max))]
    accuracy_Fedfilt_local = [np.zeros((e_max)),np.zeros((e_max)),np.zeros((e_max)),np.zeros((e_max))]
    rounds = e_max
    E =args.E # local rounds
    N_active_clients = N_clients
    N_total_clients = N_clients
    Batch_size = args.batch_size
    model_number = args.Filter_sim+1 # only four graph filters for now
    G_list = [[],[],[],[],[]] #[id, model data1, model data2, etc]
    client_list = range(N_total_clients)
    client_list = sorted(client_list, key = lambda x: random.random() )

    ## Building the models
    model_FedAvg  = CNN()
    model_Fedfilt = CNN()
    optimizer_clients = tf.keras.optimizers.SGD(learning_rate=0.02)  
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)                                                     
    model_FedAvg.compile(loss=loss,
                  optimizer=optimizer_clients,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])                  
    model_Fedfilt.compile(loss=loss,
                  optimizer=optimizer_clients,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])                                       
    global_var = model_FedAvg.get_weights()
    global_var_FedAvg = global_var
    global_var_flat = Gr.flatten(global_var)
    global_var_flat = np.array(global_var_flat)
    G = np.matlib.repmat(global_var_flat,N_active_clients,1)
    G_list[0] = client_list
    G_list[1] = G
    G_list[2] = G
    G_list[3] = G
    G_list[4] = G
    total_data_size = 0
    for i in client_list:
      client_data_size = label_test_new[i].shape[0]
      total_data_size = total_data_size + client_data_size

    ################################################################################
    ####################### Federated Learning  ####################################
    ################################################################################
    for round in range(0,rounds):
      Delta_sum = None
      print("round ",round)
      client_num = 0
      client_list = range(N_total_clients)
      client_list = sorted(client_list, key = lambda x: random.random() )
   
      for client in client_list:
        print("Client ID ",client)

        ## shuffling data manually
        client_data_size = data_train_new[client].shape[0]
        
        client_xdata = np.expand_dims(data_train_new[client],axis =-1)
        client_ydata = label_train_new[client]
    ############## FedFilt for multiple models #####################################
    ############### Local updates####################################################
        for m in range (1,model_number):
          model_Fedfilt.set_weights(Gr.unflatten(G_list[m][client],global_var))

          client_var_Fedfilt = local_update(E, model_Fedfilt,client_xdata,client_ydata,Batch_size)
          client_var_flat = Gr.flatten(client_var_Fedfilt)
          client_var_flat = np.array(client_var_flat)
          G_list[m][client,:] = client_var_flat

    ################################################################################
        model_FedAvg.set_weights(global_var_FedAvg)
        client_var_FedAvg = local_update(E, model_FedAvg,client_xdata,client_ydata,Batch_size)
        client_num += 1
    ############# FedAvg  ##########################################################
        global_var_FedAvg = server_update(method='FedAvg',delta_sum=Delta_sum, N_active_clients=N_active_clients, client_update=client_var_FedAvg, global_update=global_var_FedAvg)

    ################# Fedfilt ###################################################### 
      for m in range(1,model_number):
        if m ==1:
          G_list[m] = server_update(method='G-Fedfilt',Graph=Gph, Gradient_list=G_list[m], mu=args.mu_param)
        if m == 2:
          G_list[m] = server_update(method='G-Fedfilt',Graph=Gph, Gradient_list=G_list[m], mu=0.1)
        if m == 3:
          G_list[m] = server_update(method='G-Fedfilt',Graph=Gph, Gradient_list=G_list[m], mu=1)
        if m == 4:
          G_list[m] = server_update(method='G-Fedfilt',Graph=Gph, Gradient_list=G_list[m], mu=100)
    ################################################################################

    ########## FedFilt Evaluation ##################################################
      
      for m in range (1,model_number):
        acc_Fedfilt = np.zeros(1) # N_active_client instead of 1
        for client in range(1):
          global_var_flat = G_list[m][client,:]
          global_var = Gr.unflatten(global_var_flat,global_var)
          _,acc_Fedfilt[client] = evaluate(global_var,dataset_test,CNN(),client)
        accuracy_Fedfilt_general[m-1][round] = np.mean(acc_Fedfilt)

        acc_Fedfilt = np.zeros(N_active_clients) # N_active_client instead of 1
        for client in range(N_active_clients):
          global_var_flat = G_list[m][client,:]
          global_var = Gr.unflatten(global_var_flat,global_var)
          _,acc_Fedfilt[client] = evaluate_manual_data(global_var,data_test_new,label_test_new,CNN(),client)
        accuracy_Fedfilt_local[m-1][round] = np.mean(acc_Fedfilt)
    ################################################################################

    ########## FedAvg Evaluation ###################################################
      acc_FedAvg = np.zeros(N_active_clients)
      for client in range(N_active_clients):
        _,acc_FedAvg[client] = evaluate_manual_data(global_var_FedAvg,data_test_new,label_test_new,CNN(),client)
      accuracy_FedAvg_local[round] = np.mean(acc_FedAvg)

      acc_FedAvg = np.zeros(1)
      for client in range(1):
        _,acc_FedAvg[client] = evaluate(global_var_FedAvg,dataset_test,CNN(),client)
      accuracy_FedAvg_general[round] = np.mean(acc_FedAvg)
    ################################################################################
      print("FedAvg on general test data "+str(accuracy_FedAvg_general[round]))
      print("FedAvg on local test data "+str(accuracy_FedAvg_local[round]))
      for m in range (1,model_number):
        print("G-Fedfilt on general test data "+"model"+str(m)+str(accuracy_Fedfilt_general[m-1][round]))
      if round % 1==0:
        plt.plot(accuracy_FedAvg_general[:round])
        for m in range (1,model_number):
          plt.plot(accuracy_Fedfilt_general[m-1][:round])
        plt.title(str(N_active_clients)+" devices, "+str(E)+" local update")
        plt.xlabel("Rounds")
        plt.ylabel("General Test Accuracy")
        plt.legend(['FedAvg','G-Fedfilt1','G-Fedfilt2','G-Fedfilt3','G-Fedfilt4'])
        plt.savefig(root+"/" +args.save_path+"/Acc_VS_rounds_general.jpg", bbox_inches='tight', dpi=120)
        plt.savefig("Acc_VS_rounds_general.jpg", bbox_inches='tight', dpi=120)
        plt.close()

      for m in range (1,model_number):
        print("G-Fedfilt on local test data "+"model"+str(m)+str(accuracy_Fedfilt_local[m-1][round]))
      if round % 1==0:
        plt.plot(accuracy_FedAvg_local[:round])
        for m in range (1,model_number):
          plt.plot(accuracy_Fedfilt_local[m-1][:round])
        plt.title(str(N_active_clients)+" devices, "+str(E)+" local update")
        plt.xlabel("Rounds")
        plt.ylabel("Local Test Accuracy")
        plt.legend(['FedAvg','G-Fedfilt1','G-Fedfilt2','G-Fedfilt3','G-Fedfilt4'])
        plt.savefig(root+"/" +args.save_path+"/Acc_VS_rounds_local.jpg", bbox_inches='tight', dpi=120)
        plt.savefig("Acc_VS_rounds_local.jpg", bbox_inches='tight', dpi=120)
        plt.close()

      if rounds % 1==0:
        display.clear_output(wait=True)

if __name__ == '__main__':
   main()
