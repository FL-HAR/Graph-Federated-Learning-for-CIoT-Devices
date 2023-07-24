# Graph-Federated-Learning-for-CIoT-Devices


<p align="center">
  <img src="imgs/graph_ex.png" width="400">
   <em>image_caption</em>
</p>

<p align="center">
   <img src="imgs/example.png" width="800">
</p>

<p align="center">
  <img src="imgs/fig4.png" width="800">
</p>

## Requirements
Please install the following packages before running ``` main.py``` .
```
!pip install tensorflow-federated
!pip install pygsp==0.5.1
```
## How to run
Simply run the following code that specifies the report path, number of local updates, number of rounds, the graph used in the original paper, and label heterogeneity score.
```
python main.py --save_path 'report' --E 3  --R 400 --paper_graph 'Y' --label_hetro 5
```

# Citation
```
@article{rasti2022graph,
  title={Graph Federated Learning for CIoT Devices in Smart Home Applications},
  author={Rasti-Meymandi, Arash and Sheikholeslami, Seyed Mohammad and Abouei, Jamshid and Plataniotis, Konstantinos N},
  journal={IEEE Internet of Things Journal},
  year={2022},
  publisher={IEEE}
}
```
