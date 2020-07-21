
from utils import get_connected
import numpy as np

'''Fixed smart policy'''
def get_exp_policy(OD, args, policy_id = None):
    nb_regions = OD.shape[-1]

    if args.action_mode == 'edge':
        ones = np.ones((nb_regions,nb_regions)) 
    elif args.action_mode == 'node':
        ones = np.ones(nb_regions) 
    elif args.action_mode == 'graph':
        ones = 1
    else:
        print('Wrong action mode')
        exit()

    if policy_id is None:
        policy_id = args.expert_id

    period = args.period
    expert_h = args.expert_h
    expert_lockdown = args.expert_lockdown
    his_quotas = np.zeros(nb_regions, dtype=np.uint8) 
    expert_k =  args.expert_k
    expert_p = args.expert_p

    def act(p, pos):
        if args.action_mode == 'edge':
            p[pos] = expert_p
            p[:,pos] = expert_p
        elif args.action_mode == 'node':
            p[pos] = expert_p
        elif args.action_mode == 'graph':
            if len(pos) > 0:
                p = expert_p   
        return p


    def exp_policy(observations, p = ones):
        if args.action_mode == 'graph':
            p = 1
        else:
            p[:] = 1
        
        ps = []

        for observation in observations:
            if policy_id < 0:
                # No action
                if args.action_mode == 'graph':
                    return 1
                else:
                    return p.reshape(-1)

            OD_time = OD[observation[:period,8].astype(np.uint32)].sum(0)

            if policy_id == 0:
                # The EP-Soft policy in our paper
                SIR_v = observation[:,2:8]
                ac_m = observation[:,-2]
                ac_d = observation[:,-1]
                cond = SIR_v[:,-2] > expert_h
                cond += SIR_v[:,1] > expert_h
                if expert_lockdown > 0:
                    cond = np.bitwise_and((ac_m / (ac_d+1e-7)) < expert_lockdown, cond)
                pos = np.where(cond)[0]
                p = act(p, pos)

            # elif policy_id == 1:
            #     # Not used
            #     # Policy 1: Ban on Delta H > 0 and its connected regions
            #     # Not completed
            #     SIR_v = observation[:,2:8]
            #     times = observation[:period,8]
            #     pos = np.where(SIR_v[:,-2]> expert_h)[0]
            #     connected_ori, connected_des = get_connected(OD[times])
            #     pos_connected = []
            #     for p in pos:
            #         pos_connected = pos_connected + connected_ori[p] + connected_des[p]

            #     p = act(p, pos_connected)
        
            elif policy_id == 2:
                # The EP-Hard policy in our paper
                SIR_v = observation[:,2:8]
                ac_m = observation[:,-2]
                ac_d = observation[:,-1]
                cond = SIR_v[:,-2] > expert_h
                cond += SIR_v[:,1] > expert_h
                cond = np.bitwise_and(his_quotas < expert_k, cond)
                pos_lock = np.where(cond)[0]
                his_quotas[pos_lock] += 1
                p = act(p, pos_lock)

                his_quotas[~cond] = 0 
        
            if args.action_mode == 'edge':
                p[OD_time == 0] = 1
                ps.append(p.reshape(-1))
            elif args.action_mode == 'node':
                ps.append(p)
            elif args.action_mode == 'graph':
                ps.append([p])

        return np.array(ps)
    
    return exp_policy