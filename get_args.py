import argparse
import numpy as np

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, help='GPU',default='0')
parser.add_argument("--seed", type=int, help='random seed',default=0)

parser.add_argument("--task", type=str, help='task',default='test_list')
parser.add_argument("--verbose", type=str2bool, help='verbose',default='False')

parser.add_argument("--save_results", type=str2bool, help='save_results',default='False')
parser.add_argument("--save_path", type=str, help='save path used for test',default=None)
parser.add_argument("--save_paths", nargs='*', type=str, help='save path used for test',default=None)

# Train and Test for DDPG
parser.add_argument("--steps", type=int, help='train steps',default=400000)
parser.add_argument("--start_step", type=int, help='start step',default=0)
parser.add_argument("--warmup_steps", type=int, help='train steps',default=1000)
parser.add_argument("--optimizer", type=str, help='optimizer',default='Adam')
parser.add_argument("--lr", type=float, help='train steps',default=1e-4)
parser.add_argument("--batch_size", type=int, help='batch size',default=16)
parser.add_argument("--memory_limit", type=int, help='memory_limit',default=50000)
parser.add_argument("--train_interval", type=int, help='train_interval',default=10)
parser.add_argument("--decay", type=float, help='decay',default=0.99)
parser.add_argument("--delta_clip", type=float, help='delta_clip for loss',default=1)
parser.add_argument("--update", type=float, help='update for target models',default=0.001)

# Achitecture settings
parser.add_argument("--action_value", type=int, help='action_value',default=-1)
parser.add_argument("--action_mode", type=str, help='actions: edge/node/graph',default='edge')
parser.add_argument("--dim", type=int, help='dim',default=32)
parser.add_argument("--layer_type", type=str, help='Layer type',default='weights')
parser.add_argument("--pool_type", type=str, help='pool_type',default='flatten')

# Mobility decay
parser.add_argument("--mobility_decay", type=float, help='the weight for infections',default=0.99) 

# Reward Func
parser.add_argument("--reward_func", type=str, help='which reward func to use',default='ac_exp_weight_avg')
parser.add_argument("--base_score", type=float, help='base_score',default=100)
parser.add_argument("--NN_penal", type=float, help='NN_penal',default=100)
parser.add_argument("--total_divide", type=int, help='total_divide',default=10)

parser.add_argument("--lamda", type=float, help='the weight for infections',default=1) 
parser.add_argument("--L0", type=float, help='L0',default=72)
parser.add_argument("--H0", type=float, help='H0',default=3)

# Action noise
parser.add_argument("--action_noise", type=str2bool, help='action',default='True') 
parser.add_argument("--rd_theta", type=float, help='rd_theta',default=.15) 
parser.add_argument("--rd_sigma", type=float, help='rd_sigma',default=.3)
parser.add_argument("--rd_dt", type=float, help='rd_dt',default=1e-2) 

# Param noise
parser.add_argument("--param_noise", type=str2bool, help='param_noise',default='True')
parser.add_argument("--action_std", type=float, help='action_std',default=.1)
parser.add_argument("--init_std", type=float, help='init_std',default=0.01)
parser.add_argument("--adapt", type=float, help='adapt',default=1.01) 
parser.add_argument("--std_adapt_steps", type=int, help='std_adapt_steps',default=400000) 
parser.add_argument("--min_action_std", type=float, help='min_action_std',default=0.05) 

# Env
parser.add_argument("--repeat", type=int, help='the numebr of repeated months',default=24)
parser.add_argument("--simulation_round", type=str2bool, help='round after 50 days?',default='True')

parser.add_argument("--I_threshold", type=int, help='I_threshold',default=100)
parser.add_argument("--lockdown_threshold", type=float, help='lockdown_threshold',default=336)
parser.add_argument("--D_reward", type=str2bool, help='D_reward',default='True')
parser.add_argument("--C_reward", type=str2bool, help='C_reward',default='True')

parser.add_argument("--beta_m", type=float, help='moving infection rate',default=3) # 3
parser.add_argument("--beta_s", type=float, help='staying infection rate',default=0.1) # 0.1
parser.add_argument("--gamma", type=float, help='hospitalization rate',default=0.3) # 0.3
parser.add_argument("--theta", type=float, help='recover rate',default=0.3) # 0.3

# For fixed policy
parser.add_argument("--p", type=float, help='fixed policy',default=-1)

# For imitation learning
parser.add_argument("--expert_dif", type=str2bool, help='The reward for training comes from the difference between agent reward and expert reward',default='False')
parser.add_argument("--expert_id", type=int, help='which smart policy to use',default=0)
parser.add_argument("--expert_h", type=float, help='when expert starts lockdown',default=1)
parser.add_argument("--expert_lockdown", type=float, help='when expert ends lockdown',default= 168)
parser.add_argument("--expert_p", type=float, help='expert_p',default= 0)
parser.add_argument("--expert_k", type=int, help='expert_k',default= 5)

parser.add_argument("--prob_imitation_steps", type=int, help='prob_imitation_steps',default=200000)
parser.add_argument("--base_prob_imitation", type=float, help='base_prob_imitation',default=0.5)
parser.add_argument("--min_prob_imitation", type=float, help='min_prob_imitation',default=0)

# Init
parser.add_argument("--location_id", type=int, help='Location-id used for initialization (for testing)',default=15)
parser.add_argument("--infected_people", type=int, help='number of infected_people used for initialization (for testing)',default=20)
parser.add_argument("--rd_no_policy_days", type=int, help='how long no intervention (random)',default=0)
parser.add_argument("--fixed_no_policy_days", type=int, help='how long no intervention (fixed in env reset)',default=0)
parser.add_argument("--fixed_no_policy_days_list", nargs='*', type=int, help='how long no intervention (fixed in env reset)',default=[20])
parser.add_argument("--fixed_no_policy_i", type=float, help='start intervention from i = XX',default=0)
parser.add_argument("--fixed_no_policy_i_range", nargs='*', type=float, help='range of fixed_no_policy_i in training',default=None)


# The temporal control-resolution (Not used)
parser.add_argument("--OD_delay", type=int, help='OD_delay',default=0)
parser.add_argument("--period", type=int, help='period',default=4)


# Noise to DURLECA (Not used)
parser.add_argument("--od_obs_p", type=float, help='OD_obs_p',default=0)
parser.add_argument("--od_obs_noise", type=float, help='OD_obs_noise',default=0)
parser.add_argument("--od_misopt_p", type=float, help='OD_misopt_p',default=0)
parser.add_argument("--od_misopt_noise", type=float, help='OD_misopt_noise',default=0)

parser.add_argument("--state_obs_p", type=float, help='state_obs_p',default=0)
parser.add_argument("--state_obs_noise", type=float, help='state_obs_noise',default=0)
parser.add_argument("--state_obs_round", type=str2bool, help='round SIRH_v',default='False')
parser.add_argument("--state_import_p", type=float, help='the prob of imported cases',default=0)
parser.add_argument("--state_import_noise", type=float, help='the scale of state_import_noise',default=0)

parser.add_argument("--train_noise", type=str2bool, help='train_noise',default='False')
parser.add_argument("--test_noise", type=str2bool, help='test_noise',default='False')

parser.add_argument("--shuffle_OD", type=str2bool, help='shuffle_OD',default='False')


args = parser.parse_args()