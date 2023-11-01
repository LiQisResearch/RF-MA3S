# Quering Diverse Routes: A Deep Reinforcement Learning Approach for Multiple-Solution Traveling Salesman Problem
RF-MA3S is a solver capable of solving query route tasks such as Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP). After processing the instances, the solution set obtained exhibits a favorable balance between diversity and optimality. This solver is based on novel designs, including:

- Relativization Filter (RF): It is designed to enhance the encoder’s robustness against affine transformations (translation, rotation, scaling, and mirroring) of the input data, ingeniously embedding explicit invariance to potentially improve the quality of the discovered solutions.
- Multi-Attentive Adaptive Active Search (MA3S): It is tailored to enable the decoders to learn or mine data within different solution spaces, striking a balance between optimality and diversity.

# Dependencies




# Usage
## Training set size setting

If you want to change the training set size of the model, you need to make the following corresponding modifications to the environmental parameters in train/train.py.
```
env_params = {
    'problem_size': 20,
    'pomo_size': 20,
    'decoder_num': 5,#Multi-decoder
}
```


## Running
For training, run:
```
CUDA_VISIBLE_DEVICES=0 python train/train.py
```
For inference, run:
```
CUDA_VISIBLE_DEVICES=0 python inference/inference.py
```

For MSQI, DI, OPTI, DIFF and so on,
(1) The model that utilizes RF:
measure/main_RF.mlx
(2) Other:
measure/main_heuristic.mlx

# Acknowledgements

The code and the framework are based on [POMO](https://github.com/yd-kwon/POMO/tree/master):
```
@inproceedings{NEURIPS2020_f231f210,
 title = {POMO: Policy Optimization with Multiple Optima for Reinforcement Learning},
 author = {Kwon, Yeong-Dae and Choo, Jinho and Kim, Byoungjip and Yoon, Iljoo and Gwon, Youngjune and Min, Seungjai},
 booktitle = {Advances in Neural Information Processing Systems},
 volume = {33},
 pages = {21188--21198},
 year = {2020}
}
```

# Additional Notes
The **test data** folder contains the MSTSPLIB, TSPLIB, and CVRPLIB test sets used in the paper.
