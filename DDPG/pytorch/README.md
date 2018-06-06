# Deep Deterministic Policy Gradient [PyTorch]

### Description

DDPG algorithm framework implemented by PyTorch, the network architectures and parameters mainly follow the referenced paper, with little changes. Can solve several tasks (such as CartPole as discrete problems, and Pendulum as continuous problems) in OpenAI Gym environment, however, maybe with some steps.

### Reference

Continuous control with deep reinforcement learning (<a href="https://arxiv.org/abs/1509.02971" target="_blank">https://arxiv.org/abs/1509.02971</a>)

### Environment

- PyTorch 0.4.0
- gym 0.10.5
- tensorboardX 1.2 (need TensorFlow installed)

### Usage

```bash
python main.py --help
```

use --help to see all arguments, including environments, learning rate and so on.

Default environment is Pendulum-v0, 2 fully-connected layer with 400 and 300 neurons for Actor and Critic. Both Actor and Critic network work with batch normalization described in the paper.

