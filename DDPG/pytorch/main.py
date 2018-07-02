import os
import time
import random
import argparse
import numpy as np
import torch
import gym
from tensorboardX import SummaryWriter
from ddpg import DDPG
from normalized_env import NormalizedEnv
from train import train
from evaluator import Evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPG implemented by PyTorch')
    parser.add_argument('--env', default='Pendulum-v0', type=str, help='environment')
    parser.add_argument('--discrete', dest='discrete', action='store_true')
    parser.add_argument('--discount', default=0.99, type=float, help='bellman discount')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--memory-size', default=1000000, type=int, help='memory size')
    parser.add_argument('--action-repeat', default=3, type=int, help='repeat times for each action')

    parser.add_argument('--hidden1', default=400, type=int, help='number of first fully connected layer')
    parser.add_argument('--hidden2', default=300, type=int, help='number of second fully connected layer')
    parser.add_argument('--actor-lr', default=1e-4, type=float, help='actor learning rate')
    parser.add_argument('--critic-lr', default=1e-3, type=float, help='critic learning rate')
    parser.add_argument('--batch-size', default=64, type=int, help='minibatch size')
    parser.add_argument('--nb-train-steps', default=50, type=int, help='training times per episode')
    parser.add_argument('--max-episode-length', default=500, type=int, help='maximum episode length')

    parser.add_argument('--nb-iterations', default=2000000, type=int, help='iterations during training')
    parser.add_argument('--warmup', default=100, type=int, help='timestep without training to fill the replay buffer')
    parser.add_argument('--apply-noise', dest='apply_noise', default=True, action='store_true', help='apply noise to the action')
    parser.add_argument('--validation-interval', default=10, type=int, help='episode interval for validation')
    parser.add_argument('--checkpoint-interval', default=100, type=int, help='episode interval for checkpoint')
    parser.add_argument('--validation-episodes', default=1, type=int, help='validation episodes')

    parser.add_argument('--resume', default=None, type=str, help='resuming model path')
    parser.add_argument('--resume-num', default=-1, type=int, help='number of the weight to load')
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument('--vis', dest='vis', action='store_true')
    parser.add_argument('--cuda', dest='cuda', action='store_true')

    parser.add_argument('--seed', default=-1, type=int, help='random seed')

    args = parser.parse_args()

    # TensorBoardX summary file
    timestamp = time.time()
    timestruct = time.localtime(timestamp)
    writer = SummaryWriter(os.path.join(args.output, args.env + '@' + time.strftime('%Y-%m-%d %H-%M-%S', timestruct)))

    if args.discrete:
        env = gym.make(args.env)
        env = env.unwrapped
    else:
        env = NormalizedEnv(gym.make(args.env))

    # set random seed
    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        env.seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    # states and actions space
    nb_states = env.observation_space.shape[0]
    if args.discrete:
        nb_actions = env.action_space.n
    else:
        nb_actions = env.action_space.shape[0]

    evaluator = Evaluator(args)

    agent = DDPG(nb_states, nb_actions, args)

    # resume train
    if args.resume is not None and args.resume_num is not -1:
        print('resume train, load weight file: {}...'.format(args.resume_num))
        agent.load_model(args.output, args.resume_num)

    train(agent, env, evaluator, writer, args)
