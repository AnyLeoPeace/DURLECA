import numpy as np

'''Reward Func'''
def get_reward_func_dict(args):

    print('Using reward func',args.reward_func)
    
    def ac_power_exp_reward_func(SIRH, OD, current_m, accumlated_m, accumlated_d):
        h = SIRH[:,3].mean()
        accumlated_ratio = accumlated_m / (accumlated_d + 1e-7)

        ri = args.lamda * (np.exp(h/args.H0) - 1) # norm to 0~1 (when h < H0)
        rm = np.power(accumlated_ratio, args.L0).mean() # norm to 0~1

        r = (args.base_score + rm - ri)/args.total_divide
        
        return r
    

    def ac_exp_reward_func(SIRH, OD, current_m, accumlated_m, accumlated_d):
        # Note, here we count mobility as a cost. And accumlated_d is fixed as the mean out at each region.
        h = SIRH[:,3].mean()
        accumlated_ratio = accumlated_m / (accumlated_d + 1e-7) / args.L0 

        ri = args.lamda * (np.exp(h/args.H0) - 1)# norm to 0~1 (when h < H0)
        rm = 1 - np.exp(accumlated_ratio).mean() # norm to 0~1

        r = (args.base_score + rm - ri)/args.total_divide

        return r
    
    def ac_exp_weight_avg_reward_func(SIRH, OD, current_m, accumlated_m, accumlated_d):
        # Note, here we count mobility as a cost. And accumlated_d is fixed as the mean out at each region.
        h = SIRH[:,3].mean()
        current_lost = OD.sum(0).sum(-1) - current_m.sum(0).sum(-1)
        current_ratio = current_lost[:,np.newaxis] / (accumlated_d + 1e-7)
        accumlated_ratio = (accumlated_m - current_lost[:,np.newaxis]) / args.mobility_decay / (accumlated_d + 1e-7) / args.L0 
        ri = args.lamda * (np.exp(h/args.H0) - 1)# norm to 0~1 (when h < H0)
        rm = - (np.exp(accumlated_ratio) * current_ratio).mean() # norm to 0~1

        r = (args.base_score + rm - ri)/args.total_divide

        return r

    def ac_power_weight_avg_reward_func(SIRH, OD, current_m, accumlated_m, accumlated_d):
        # Note, here we count mobility as a cost. And accumlated_d is fixed as the mean out at each region.
        h = SIRH[:,3].mean()
        current_lost = OD.sum(0).sum(-1) - current_m.sum(0).sum(-1)
        current_ratio = current_lost[:,np.newaxis] / (accumlated_d + 1e-7)
        accumlated_ratio = (accumlated_m - current_lost[:,np.newaxis]) / args.mobility_decay / (accumlated_d + 1e-7) / args.L0 

        ri = args.lamda * (np.exp(h/args.H0) - 1)# norm to 0~1 (when h < H0)
        rm = - (np.power(accumlated_ratio, 2) * current_ratio).mean() # norm to 0~1

        r = (args.base_score + rm - ri)/args.total_divide

        return r


    def ac_exp_weight_norm_reward_func(SIRH, OD, current_m, accumlated_m, accumlated_d):
        # Note, here we count mobility as a cost. And accumlated_d is fixed as the mean out at each region.
        OD_all = OD.sum(0).sum(-1)
        h = SIRH[:,3].mean()
        current_lost = OD_all - current_m.sum(0).sum(-1)
        current_ratio = current_lost[:,np.newaxis] / (OD_all + 1e-7)
        accumlated_ratio = (accumlated_m - current_lost[:,np.newaxis])/ args.mobility_decay / (accumlated_d + 1e-7) / args.L0
        
        ri = args.lamda * (np.exp(h/args.H0) - 1)# norm to 0~1 (when h < H0)
        rm = - (np.exp(accumlated_ratio) * current_ratio).mean() # norm to 0~1

        r = (args.base_score + rm - ri)/args.total_divide

        return r
    
    def ac_power_weight_norm_reward_func(SIRH, OD, current_m, accumlated_m, accumlated_d):
        # Note, here we count mobility as a cost. And accumlated_d is fixed as the mean out at each region.
        OD_all = OD.sum(0).sum(-1)
        h = SIRH[:,3].mean()
        current_lost = OD_all - current_m.sum(0).sum(-1)
        current_ratio = current_lost[:,np.newaxis] / (OD_all + 1e-7)
        accumlated_ratio = (accumlated_m - current_lost[:,np.newaxis])/ args.mobility_decay / (accumlated_d + 1e-7) / args.L0
        
        ri = args.lamda * (np.exp(h/args.H0) - 1)# norm to 0~1 (when h < H0)
        rm = - (np.power(accumlated_ratio, 2) * current_ratio).mean() # norm to 0~1

        r = (args.base_score + rm - ri)/args.total_divide

        return r
        
    reward_func_dict = {
        'ac_exp_norm':ac_exp_reward_func,
        'ac_power_exp_norm':ac_power_exp_reward_func,
        'ac_exp_weight_avg': ac_exp_weight_avg_reward_func,
        'ac_power_weight_norm': ac_power_weight_norm_reward_func,
        'ac_power_weight_avg': ac_power_weight_avg_reward_func,
        'ac_exp_weight_norm': ac_exp_weight_norm_reward_func,
    }

    return reward_func_dict

