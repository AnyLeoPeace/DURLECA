import json
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

def plot_results(path, counts, ODs, ODs_daily, name = None):
    # Plot OD and infections
    f = plt.figure(figsize = (15,10))
    plt.subplot(2,2,1)
    plt.plot(range(len(ODs)), ODs, label='RL')
    plt.legend(loc='upper right')
    plt.ylabel('Mobility')
    plt.xlabel('Time steps')
    # 
    plt.subplot(2,2,2)
    plt.plot(range(len(ODs_daily)), ODs_daily, label='RL')
    plt.legend(loc='upper right')
    plt.ylabel('Mobility (Daily Mean)')
    plt.xlabel('Time steps')
    # 
    plt.subplot(2,2,3)
    plt.plot(range(len(counts)), counts[:, 1], label= 'RL')
    plt.legend(loc='upper right')
    plt.ylabel('Infection')
    plt.xlabel('Time steps')
    # 
    plt.subplot(2,2,4)
    plt.plot(range(len(counts)), counts[:, 3], label= 'RL')
    plt.legend(loc='upper right')
    plt.ylabel('Hospitalized')
    plt.xlabel('Time steps')
    if name is None:
        f.savefig(path + '/results.png')
    else:
        f.savefig(path + '/' + name + '_results.png')
    plt.close('all')


# Plot his
def plot_training(path, name = None):
    if os.path.exists(path + '/his'):
        with open(path + '/his', 'rb') as f:
            his = pickle.load(f)

        f = plt.figure(figsize = (14,4))
        plt.subplot(1,3,1)
        plt.plot(his['nb_steps'], his['episode_reward'])
        plt.ylabel('episode reward')
        plt.xlabel('steps')

        plt.subplot(1,3,2)
        plt.plot(his['nb_steps'], his['nb_episode_steps'])
        plt.ylabel('number of episode steps')
        plt.xlabel('steps')

        plt.subplot(1,3,3)
        plt.plot(his['nb_steps'], np.array(his['episode_reward']) / np.array(his['nb_episode_steps']))
        plt.ylabel('mean reward')
        plt.xlabel('steps')

        if name is None:
            f.savefig(path + '/training.png')
        else:
            f.savefig(path + '/' + name + '_training.png')
        
        plt.close('all')

    # Plot Log
    if os.path.exists(path + '/log'):
        with open(path+'log', 'r') as f:
            data = json.load(f)
        
        keys = ['loss', 'mae', 'mean_q', 'episode_reward', 'nb_episode_steps', 'nb_steps']
        f = plt.figure(figsize=(10,6))
        for i in range(5):
            key = keys[i]
            plt.subplot(2,3,i+1)
            plt.plot(data['nb_steps'], data[key], label = key)
            plt.legend()

        plt.suptitle('Training Log')

        if name is None:
            f.savefig(path + '/log.png')
        else:
            f.savefig(path + '/' + name + '_log.png')


    
if __name__ == '__main__':
    from get_args import args
    plot_training(args.save_path)