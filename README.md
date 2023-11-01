# Quering Diverse Routes: A Deep Reinforcement Learning Approach for Multiple-Solution Traveling Salesman Problem
RF-MA3S is a solver capable of solving query route tasks such as Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP). After processing the instances, the solution set obtained exhibits a favorable balance between diversity and optimality. This solver is based on novel designs, including:

- Relativization Filter (RF): It is designed to enhance the encoder’s robustness against affine transformations (translation, rotation, scaling, and mirroring) of the input data, ingeniously embedding explicit invariance to potentially improve the quality of the discovered solutions.
- MultiAttentive Adaptive Active Search (MA3S): It is tailored to enable the decoders to learn or mine data within different solution spaces, striking a balance between optimality and diversity.

# Dependencies




# Usage

For training, run:
```
CUDA_VISIBLE_DEVICES=0 python train/train.py
```
For inference, run:
```
CUDA_VISIBLE_DEVICES=0 python inference/inference.py
```

For MSQI, DI, OPTI, DIFF and so on,
(1) RF-MA3S:
measure/main_RF.mlx
(2) Other:
measure/main_heuristic.mlx

# Acknowledgements


# Additional Notes
The test data folder contains the MSTSPLIB, TSPLIB, and CVRPLIB test sets used in the paper.
