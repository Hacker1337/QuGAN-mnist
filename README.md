The following code is used to replicate the results of the paper:

Qugan: A generative adversarial network through quantum states, Samuel A. Stein and Betis Baheri and Daniel Chen and Ying Mao and Qiang Guan and Ang Li and Bo Fang and Shuai Xu, 2021 IEEE International Conference on Quantum Computing and Engineering (QCE).

Fit classical model

`python pennylane-torch-implementation.py --epoch 100 --model c --g_lr 1e-4 --d_lr 1e-3`