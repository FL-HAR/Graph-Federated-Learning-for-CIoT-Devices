# !pip install pygsp==0.5.1

from pygsp import graphs, filters, plotting
import collections
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

root = os.getcwd()


def flatten(weights):
  """
Use to flatten the weights of the network
  """
  w = []
  for l in weights:
    if isinstance(l,collections.Iterable):
      w = w + flatten(l)
    else:
      w = w + [l]
  return w

def unflatten(flat_w, old_w):
  """
Use to unflatten the weights of the network
  """
# """ flat_w : 1D array of weights (flattened).old_w : output of model.get_weights(). """
  new_w = [ ]
  
  i = 0
  for layer in old_w:
    size = layer.size
    new_w.append(flat_w[i:i+size].reshape(layer.shape))
    i += size

  return new_w 

def graph_ex(N_clients):

    """
    An example of a graph 
      """
  
    plotting.BACKEND = 'matplotlib'
    plt.rcParams['figure.figsize'] = (10, 5)
    rs = np.random.RandomState(42)

    Gph = graphs.Community(N=N_clients,Nc=4,min_comm=5,)

    Gph.compute_fourier_basis()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for i, ax in enumerate(axes):
        Gph.plot_signal(Gph.U[:, i], vertex_size=30, ax=ax)
        _ = ax.set_title('Eigenvector {}'.format(i))
        ax.set_axis_off()
    fig.tight_layout()
    return Gph

def graph_in_the_paper():
  """
Returns the graph used in the evaluation of the paper https://doi.org/10.1109/JIOT.2022.3228727 
  """
  t = sio.loadmat(root+"/graph_info/w")
  w = t['w']
  t = sio.loadmat(root+"/graph_info/coords")
  coords = t['coords']
  Gg = graphs.Graph(w,coords=coords)
  Gph = Gg
  Gph.compute_fourier_basis()
  Gph.compute_laplacian()
  fig, axes = plt.subplots(1, 2, figsize=(10, 3))
  for i, ax in enumerate(axes):
      Gph.plot_signal(Gg.U[:, i], vertex_size=30, ax=ax)
      _ = ax.set_title('Eigenvector {}'.format(i))
      ax.set_axis_off()
  fig.tight_layout()
  return Gph

def hard_filt(signal,Gph,func,mu_h):
  """
filter the signal on the graph using hard thresholding and parameter mu_h
  """
  f_hat = Gph.gft(signal)
  filt_coeff= func(Gph.e,mu_h,)
  f_hat = np.matmul(np.diag(filt_coeff),f_hat)
  return Gph.igft(f_hat)
  
def soft_filt(signal,Gph,func,mu_s1,mu_s2):
  """
filter the signal on the graph using hard thresholding and parameter mu_h
  """
  f_hat = Gph.gft(signal)

  # for i in range(f_hat.shape[1]):
  filt_coeff= func(Gph.e,mu_s1,mu_s2)
  f_hat = np.matmul(np.diag(filt_coeff),f_hat)
  return Gph.igft(f_hat)

def h_s(x,mu_s1,mu_s2):
    return (1-mu_s2) / (1. + mu_s1 * x) +mu_s2

def h_h(x,mu_h):
  cnt =0
  y = np.zeros_like(x)
  for i in x:
    if i<mu_h:
      y[cnt] = 1
    else:
      y[cnt] = 0
    cnt += 1
  return y

def plot_gfilter(func,mu_s1,mu_s2,Graf):
  from numpy.core.function_base import linspace
  lamda = linspace(0,np.max(Graf.e)+2,200)
  if mu_s2 != None:
    f_respond = func(lamda,mu_s1,mu_s2)
  else:
    f_respond = func(lamda,mu_s1)
  plt.plot(lamda,f_respond,color='black',linewidth='5')
  
  for i in range(0,50):
    plt.scatter(Graf.e,np.zeros_like(Graf.e)+i/50,marker='|',color='#B4B4B4',linewidths='.001')
    plt.savefig('filter_0',dpi=200)
  # plt.title("Filter Frequency Response")

def graph_ex(N_clients=30):

    """
    An example of a graph 
      """
    font = {
        'size'   : 10}
    plt.rc('font', **font)

    plotting.BACKEND = 'matplotlib'
    plt.rcParams['figure.figsize'] = (10, 5)
    rs = np.random.RandomState(42)

    Gph = graphs.Sensor(N=N_clients)

    Gph.compute_fourier_basis()
    print(30*"#"+" the first and second eigenvalues of the graph "+ 30*"#")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for i, ax in enumerate(axes):
        Gph.plot_signal(Gph.U[:, i], vertex_size=30, ax=ax)
        _ = ax.set_title('Eigenvector {}'.format(i))
        ax.set_axis_off()
    fig.tight_layout()
    plt.pause(1)
    print(2*("|"+100*" "+"|\n"))
    print(30*"#"+" the graph filter with parameters 1 and 0 "+ 30*"#")
    font = {
        'size'   : 25}
    plt.rc('font', **font)
    plot_gfilter(h_s,10,0,Gph)
    font = {
        'size'   : 10}
    plt.rc('font', **font)
    plt.pause(1)
    print(30*"#"+" creating random signal on the graph "+ 30*"#")
    rs = np.random.RandomState(42)
    s = np.zeros((Gph.N,Gph.N))
    s += 1.5*rs.uniform(-0.5, 0.5, size=(Gph.N,Gph.N))+ np.random.rand(Gph.N,Gph.N)
    f = soft_filt(s,Gph,h_s,10,0)
    # f= soft_filt(s,h_s,100000,0)

    id=1
    MEAN = np.mean(s[:,id])
    MEAN_after = np.mean(f[:,id])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    # print(2*("|"+100*" "+"|\n"))
    print(30*"#"+" filtered signal before and after graph filtering "+ 30*"#")
    Gph.plot_signal(s[:,id], vertex_size=30, ax=axes[0])
    _ = axes[0].set_title('before aggregation, average = '+str((MEAN)))
    axes[0].set_axis_off()
    Gph.plot_signal(f[:,id], vertex_size=30, ax=axes[1])
    _ = axes[1].set_title('aggregated gradient average =' +str((MEAN_after)))
    axes[1].set_axis_off()
    fig.tight_layout()

    return Gph