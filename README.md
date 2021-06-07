# Policy gradient algorithms on OpenAI's Cartpole

## Summary
&nbsp;&nbsp;&nbsp;&nbsp;The goal of this application is to implement Policy Gradient algorithms on Open AI Cartpole enviroment. Algorithms that were implemented include: 
  * [REINFORCE](https://link.springer.com/article/10.1007/BF00992696)
  * [Actor-Critic (AC)](https://ieeexplore.ieee.org/abstract/document/6313077)
  * [A2C](https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py)
  * [A3C](https://arxiv.org/pdf/1602.01783.pdf)

&nbsp;&nbsp;&nbsp;&nbsp;Results show difference in efficiency between REINFORCE and Actor-Critic algorithm as well as between A2C and A3C algorithms. Algorithms were compared based on whether algorithm uses multiprocessing or not. As it can be seen on the graph below, Actor-Critic achieves considerable improvement in efficiency over REINFORCE, while A3C shows some improvement comparing to A2C.

## Environment


## Policy Gradient

## Structure

## Algo 1

## Algo 2

## Algo 3

## Algo 4

## Results

    
## Rest of the data and TensorBoard
&nbsp;&nbsp;&nbsp;&nbsp;Rest of the training data can be found at [/content/runs](https://github.com/leonjovanovic/deep-reinforcement-learning-atari-pong/tree/main/content/runs). If you wish to see it and compare it with the rest, I recommend using TensorBoard. After installation simply change the directory where the data is stored, use the following command
  
```python
LOG_DIR = "full\path\to\data"
tensorboard --logdir=LOG_DIR --host=127.0.0.1
```
and open http://localhost:6006 in your browser.
For information about installation and further questions visit [TensorBoard github](https://github.com/tensorflow/tensorboard/blob/master/README.md)

## Future improvements
  


