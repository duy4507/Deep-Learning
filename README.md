# Deep-Learning

2022-2023 M2 IASD Deep Learning

Building a Neural Network to play Go with 100k parameters limit.

## Architecture

The network outputs targets are the policy (a vector of size 361 with 1.0 for the move played, 0.0 for the other moves), and the value (close to 1.0 if White wins, close to 0.0 if Black wins). It uses the ResNet architecture combined with MixNet. The learning rate decays using the Cosine Annealing formula. This work was mainly inspired by this [paper](https://www.lamsade.dauphine.fr/~cazenave/papers/MobileNetworksForComputerGo.pdf) published by [Tristan Cazenave](https://www.lamsade.dauphine.fr/~cazenave/index.php) in 2022.

The model is at 4625.h5 which is the validation score for the model after 1 000 epochs of training.

## Data

This was trained using a training set with 1 000 000 different games in total. The input data is composed of 31 19x19 planes (color to play, ladders, current state on two planes, two previous states on four planes).


