from get_args import args

import random
import os
import pickle
import matplotlib.pyplot as plt
import time
import sys
import json
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import tensorflow as tf
np.random.seed(args.seed)
random.seed(args.seed)
tf.random.set_seed(args.seed)

tf.compat.v1.disable_eager_execution()
physical_devices = tf.config.list_physical_devices('GPU') 
try: 
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
except: 
  pass 

from env_SIRH import COVID_Env
from model_build import build_agent
from exp_policy import get_exp_policy
from reward_func import get_reward_func_dict
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from utils import *
from plot import *




localtime = time.strftime("%m%d_%H:%M:%S", time.localtime()) 

# Episode end (succeed) reward
def C_func(remain_steps):
    return (1 - np.power(args.decay, remain_steps)) / (1-args.decay)

C_dict = {
    True: C_func,
    False: None
}

# Episode end (failure) reward
def D_func(remain_steps):
    return (1 - np.power(args.decay, remain_steps)) / (1-args.decay)

D_dict = {
    True: D_func,
    False: None
}

def get_max_start_func(args):

    if args.fixed_no_policy_i_range is None:
        return None
    else:
        print('Using random i start',args.fixed_no_policy_i_range)
        class Random_i():
            def __init__(self, args):
                self.low = args.fixed_no_policy_i_range[0]
                self.high = args.fixed_no_policy_i_range[1]
                self.d = np.random.uniform(low = self.low, high = self.high)
            def max_start_func(self, state):
                if state[:,1].mean() > self.d:
                    print('Random Start at I =', state[:,1].mean())
                    self.d = np.random.uniform(low = self.low, high = self.high)
                    return True
                else:
                    return False

        random_i = Random_i(args)
        return random_i.max_start_func

def get_OD_tensor(OD, args):
    OD_d = np.copy(OD)
    a,b,c = OD.shape
    if args.od_obs_p > 0 and args.od_obs_noise > 0:
        print('Use od obs noise')
        # np.random.seed(args.seed)
        od_obs_flag = np.bitwise_and(np.random.rand(a,b,c) < args.od_obs_p, OD_d > 0) 
        OD_d_n = np.int32(OD_d * od_obs_flag * np.random.randn(a,b,c) * args.od_obs_noise)
        OD_d = np.int32(OD_d) + OD_d_n
        OD_d[OD_d<0] = 0 
    
    return tf.constant(OD_d, dtype=tf.float32)
    

def train(args):
    '''Set env'''
    OD, population = load_data()
    if args.OD_delay > 0:
        print('OD delay',args.OD_delay)
        OD = np.vstack([OD[args.OD_delay:],OD[:args.OD_delay]])

    nb_regions = OD.shape[-1]

    betas_s = np.ones(nb_regions) * (args.beta_s / 24) 
    betas_m = np.ones(nb_regions) * (args.beta_m / 24) 
    gammas = np.ones(nb_regions) * (args.gamma / 24)  
    thetas = np.ones(nb_regions) * (args.theta / 24) 

    reward_func_dict = get_reward_func_dict(args)
    expert_dif = get_exp_policy(OD, args) if args.expert_dif == True else None
    env = COVID_Env(
            population, OD, reward_func_dict[args.reward_func], betas_m, betas_s, gammas, thetas, \
            fixed_no_policy_days=args.fixed_no_policy_days, fixed_no_policy_i=0, \
            C_reward_func = C_dict[args.C_reward], D_reward_func = D_dict[args.D_reward], reward_clip = np.array([-args.NN_penal, args.base_score])/args.total_divide,\
            period = args.period, total_time=744*args.repeat, I_threshold = args.I_threshold, lockdown_threshold=args.lockdown_threshold, \
            simulation_round = args.simulation_round,  mobility_decay = args.mobility_decay,  expert_dif = expert_dif,
            state_import_p = args.state_import_p, state_import_noise = args.state_import_noise, \
            state_obs_p = args.state_obs_p, state_obs_noise = args.state_obs_noise, state_obs_round = args.state_obs_round, \
            od_obs_p = args.od_obs_p, od_obs_noise = args.od_obs_noise, od_misopt_p = args.od_misopt_p, od_misopt_noise = args.od_misopt_noise, \
            shuffle_OD = args.shuffle_OD)
    
    if args.train_noise == False:
        env.set_no_noise()

    '''Build model'''
    OD_tensor = get_OD_tensor(OD, args)
    agent = build_agent(OD, OD_tensor, args)

    if args.save_path is not None:
        print('Load saved agent at',args.save_path)
        agent.load_weights(args.save_path + 'agent_weights.ckpt')

    path = './save/' + localtime + '/'
    os.mkdir(path)
    print('-'*20,'Save at', path,'-'*20)

    save_path = path + 'agent_weights.ckpt'
    if args.steps > 0:
        his = agent.fit(
            env, nb_steps=args.steps, 
            callbacks = [
                ModelIntervalCheckpoint(save_path, interval = 10000, verbose = 1),
                FileLogger(path + '/log', interval = 1),
            ], 
            visualize=False, verbose=2, max_start_func = get_max_start_func(args),
            nb_max_start_steps=args.rd_no_policy_days * 24 //args.period, start_step_policy = get_exp_policy(OD, args, policy_id = -1) # Here means no policy days
            )
    

    print('Start Testing')
    args.save_path = path
    test(env, OD, agent.select_action, path, args)

    with open(path + '/commandline_args.txt', 'w') as f:
        f.write('\n'.join(sys.argv[1:]))
    
    json_str = json.dumps(vars(args))
    with open(path+ '/params.json', 'w') as json_file:
        json_file.write(json_str)

    with open(path + '/his', 'wb') as f:
        pickle.dump(his.history, f)

    agent.save_weights(save_path, overwrite=True)
    print('-'*20,'Save at', save_path,'-'*20)


def test_list(args):
    '''Set env'''
    OD, population = load_data()

    if args.OD_delay > 0:
        print('OD delay',args.OD_delay)
        OD = np.vstack([OD[args.OD_delay:],OD[:args.OD_delay]])
    
    nb_regions = OD.shape[-1]
    
    betas_s = np.ones(nb_regions) * (args.beta_s / 24) 
    betas_m = np.ones(nb_regions) * (args.beta_m / 24) 
    gammas = np.ones(nb_regions) * (args.gamma / 24)  
    thetas = np.ones(nb_regions) * (args.theta / 24) 

    reward_func_dict = get_reward_func_dict(args)
    expert_dif = get_exp_policy(OD, args) if args.expert_dif == True else None
    env = COVID_Env(
            population, OD, reward_func_dict[args.reward_func], betas_m, betas_s, gammas, thetas, \
            fixed_no_policy_days=args.fixed_no_policy_days, fixed_no_policy_i=0, \
            C_reward_func = C_dict[args.C_reward], D_reward_func = D_dict[args.D_reward], reward_clip = np.array([-args.NN_penal, args.base_score])/args.total_divide,\
            period = args.period, total_time=744*args.repeat, I_threshold = args.I_threshold, lockdown_threshold=args.lockdown_threshold, \
            simulation_round = args.simulation_round,  mobility_decay = args.mobility_decay,  expert_dif = expert_dif,
            state_import_p = args.state_import_p, state_import_noise = args.state_import_noise, \
            state_obs_p = args.state_obs_p, state_obs_noise = args.state_obs_noise, state_obs_round = args.state_obs_round, \
            od_obs_p = args.od_obs_p, od_obs_noise = args.od_obs_noise, od_misopt_p = args.od_misopt_p, od_misopt_noise = args.od_misopt_noise,
            shuffle_OD = args.shuffle_OD)

    print('-'*30)
    if args.save_path is not None:
        '''Build model'''
        print('Use trained agent at', args.save_path)
        OD_tensor = get_OD_tensor(OD, args)
        # OD_tensor = tf.constant(OD, dtype=tf.float32)
        agent = build_agent(OD, OD_tensor, args)
        agent.load_weights(args.save_path+ 'agent_weights.ckpt')
        select_action = agent.select_action
        path = args.save_path
        test(env, OD, select_action, path, args)
    
    elif args.save_paths is not None:
        OD_tensor = get_OD_tensor(OD, args)
        agent = build_agent(OD, OD_tensor, args)
        for save_path in args.save_paths:
            print('Use trained agent at', save_path)
            agent.load_weights(save_path+ 'agent_weights.ckpt')
            select_action = agent.select_action
            path = save_path
            test(env, OD, select_action, path, args)
    
    elif args.p >= 0:
        ones = np.ones((nb_regions*nb_regions,))
        print('Use fixed policy',args.p)
        select_action = lambda x: ones * args.p
        path = 'save/fixed_' + str(args.p) + '/'
        if os.path.exists(path) == False:
            os.mkdir(path)
        test(env, OD, select_action, path, args)
    else:
        print('Use Smart Policy')
        select_action = get_exp_policy(OD, args)
        path = 'save/expert_id_' + str(args.expert_id) + '_h' + str(args.expert_h) + '_l' + str(args.expert_lockdown) + '_p' + str(args.expert_p) + '_k' + str(args.expert_k) + '/'
        if os.path.exists(path) == False:
            os.mkdir(path)
        test(env, OD, select_action, path, args)


def test(env, OD, select_action, path, args):
    rewards = []
    all_results = {}
    base_score = args.base_score//args.total_divide

    env.expert_dif = None
    env.I_threshold = 1000
    env.lockdown_threshold = -1
    no_policy_flag = True
    nb_regions = OD.shape[-1]

    for no_days in args.fixed_no_policy_days_list:
        env.fixed_no_policy_days = 0
        env.fixed_no_policy_i = 0
        env.set_no_noise()
        test_noise = args.test_noise
        print('Set fixed_no_policy_days as',no_days)

        state = env.reset(args.location_id, args.infected_people)
        reward = 0
        counts = []
        actions = []
        actions_ = []
        
        if args.action_mode == 'edge':
            p = np.ones((nb_regions*nb_regions,))
        elif args.action_mode == 'node':
            p = np.ones((nb_regions,))
        else:
            p = 1
        states = [state[:,:4]]

        done = False
        ODs = []
        ODs_daily = []
        ODs_origin = []
        ODs_origin_daily = []
        while done is False:
            i = env.time_step 

            if args.fixed_no_policy_i > 0:
                # Rely on args.start_h
                if state[:,1].mean() < args.fixed_no_policy_i and no_policy_flag == True:
                    action = p
                    # action_ = p
                else:
                    no_policy_flag = False
                    if args.verbose == True:
                        print('Using Select Action', (i+ env.accumulated_time) // 24)
                    action = select_action([env.obs_state])
                    # action_ = select_action([env.state])
            else:
                if (i+ env.accumulated_time) // 24 < no_days:
                    action = p
                    # action_ = p
                else:
                    if test_noise == True:
                        env.set_noise()
                        test_noise = False
                    if args.verbose == True:
                        print('Using Select Action', (i+ env.accumulated_time) // 24)
                    action = select_action([env.obs_state])
                    # action_ = select_action([env.state])
            
            actions.append(action)
            # actions_.append(action_)

            state, r, done, _ = env.step(action)
            reward += r
            count = env.counts()
            counts.append(count)
        
            OD_sum = OD[i:i+args.period].sum(0)
            if args.action_mode == 'edge':
                action = action.reshape((nb_regions,nb_regions))
                # action_ = action_.reshape((nb_regions,nb_regions))
            elif args.action_mode == 'node':
                action = action.reshape((nb_regions,1))
                # action_ = action_.reshape((nb_regions,1))

            OD_p = OD_sum * action
            # OD_p_ = OD_sum * action_
            ODs.append(OD_p)
            ODs_origin.append(OD_sum)
        
            if args.verbose == True:
                print('Period',len(counts), 'SIR', count / nb_regions,'; M', OD_p.mean()/OD_sum.mean())
                # print('Period',len(counts), 'SIR', count / nb_regions,'; M', OD_p.mean()/OD_sum.mean(), 'Right M', OD_p_.mean()/OD_sum.mean())
                print('Reward',r)
                print('-'*30)
            

        print('No days', no_days, '; Reward',reward)

        no_policy_len = no_days * 24 // args.period
        counts = np.array(counts)
        ODs = np.array(ODs)[no_policy_len:]
        ODs_origin = np.array(ODs_origin)[no_policy_len:]
        rewards.append(reward)

        len_d = 24 // args.period
        days = len(ODs) // len_d
        ODs_daily = ODs[:days*len_d].reshape(-1, len_d, nb_regions, nb_regions).sum(1)
        ODs_origin_daily = ODs_origin[:days*len_d].reshape(-1, len_d, nb_regions, nb_regions).sum(1)

        results = {}
        results['start_intervene'] = no_days
        results['reward'] = reward
        results['reward_'] = reward - ODs.shape[0]*base_score - base_score*100
        results['SIRH'] = np.array(counts)
        results['ODs_out'] = ODs.sum(-1)
        results['ODs_origin_out'] = ODs_origin.sum(-1)
        results['pandemic_duration'] = len(ODs) / len_d + no_days
        results['path'] = path

        if args.save_results == True:
            np.save(path + './results_dict_' + str(no_days), results)

        results = get_metrics(results, ODs, ODs_origin, ODs_daily, ODs_origin_daily)
        all_results[no_days] = results

    if args.save_path is not None:
        plot_training(path)
    
    for no_days in all_results:
        results = all_results[no_days]
        print('Start intervention',results['start_intervene'],'-'*30)
        print('Pandemic Duration',results['pandemic_duration'])
        print('Reward',results['reward'])
        print('Reward_',results['reward_'])

        print('Mean H',results['mean_h'])
        print('Max H',results['max_h'])
        print('Total R',results['total_r'])

        print('Mean M', results['ODs_out'].mean() / results['ODs_origin_out'].mean())
        print('Max M', results['city_daily_ratio'].max())

        print('City Lockdown Duration',results['city_lockdown_duration'])
        print('Strictest Region Lockdown Duration',results['strict_region_lockdown_duration'])
        print('Strictest Region Mobility Mean',results['strict_region_total_ratio'])
        print('-'*30)



def get_metrics(results, ODs, ODs_origin, ODs_daily, ODs_origin_daily):
    # mean/max h
    nb_regions = ODs.shape[-1]

    results['mean_h'] = np.mean(results['SIRH'][:,-1]) 
    results['max_h'] = np.max(results['SIRH'][:,-1])
    results['total_r'] = results['SIRH'][-1,-2]

    results['city_period_ratio'] = ODs.mean(axis=(-1,-2)) / (ODs_origin.mean(axis=(-1,-2)) + 1e-7)
    results['city_daily_ratio'] = ODs_daily.mean(axis=(-1,-2)) / (ODs_origin_daily.mean(axis=(-1,-2)) + 1e-7)

    results['region_period_ratio'] = ODs.mean(axis=-1) / (ODs_origin.mean(axis=-1) + 1e-7)
    results['region_daily_ratio'] = ODs_daily.mean(axis=-1) / (ODs_origin_daily.mean(axis=-1) + 1e-7)

    # The lockdown duration
    city_lockdown = results['city_daily_ratio'] < 0.2 # D
    results['city_lockdown_duration'] = np.sum(city_lockdown)

    # The most restricted area
    region_ratio = ODs_daily.mean(axis=(-1,0)) / ODs_origin_daily.mean(axis=(-1,0)) # 
    strict_region_id = np.argmin(region_ratio)
    results['strict_region_id'] = strict_region_id
    results['strict_region_daily_ratio'] = results['region_daily_ratio'][:, strict_region_id]
    results['strict_region_lockdown_duration'] = np.sum(results['strict_region_daily_ratio'] < 0.2)
    results['strict_region_total_ratio'] = region_ratio[strict_region_id]

    plot_results(results['path'], results['SIRH'] / nb_regions, results['city_period_ratio'], results['city_daily_ratio'], name = str(results['start_intervene']))

    return results

if __name__ == '__main__':
    eval(args.task)(args)
