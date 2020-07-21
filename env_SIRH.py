import gym
import numpy as np
from env_func_SIRH import *
from gym import spaces
import numpy as np

def round_func(x):
    # 只向下取整
    x = np.where(x - np.floor(x) < 0.01, np.floor(x), x)
    x[x<0] = 0
    return x

def state_obs_round(x):
    rd_x = np.random.rand(size=x)
    p_x = x - np.floor(x)
    return np.where(p_x > rd_x, np.floor(x), np.ceil(x))



class COVID_Env(gym.Env):
     
    def __init__(self, population, OD, reward_func, betas_m, betas_s, gammas, thetas, C_reward_func, D_reward_func = None,
                fixed_no_policy_days = 0, fixed_no_policy_i = 0, mobility_decay = 0, I_threshold = 100, lockdown_threshold = -1,
                period = 24, total_time=744, simulation_round = False, expert_dif = None, reward_clip = [0,100],
                state_import_p = 0, state_import_noise = 0, state_obs_p = 0, state_obs_noise = 0, state_obs_round = False,
                od_misopt_p = 0, od_misopt_noise = 0, od_obs_p = 0, od_obs_noise = 0,
                shuffle_OD = False, seed = 0):

        print('-'*30,'Build Env','-'*30)
        self.population = population
        self.population_sum = population.sum()
        self.action_space = None
        self.observation_space = None
        self.betas_m = betas_m
        self.betas_s = betas_s
        self.gammas = gammas
        self.thetas = thetas
        self.accumulated_time = 0
        self.time_step = 0
        self.total_time = total_time
        self.nb_regions = OD.shape[-1]
        self.OD = OD
        self.action_space = spaces.Box(low=0,high=1, shape=(self.nb_regions,self.nb_regions))
        self.period = period
        self.reward_func = reward_func
        self.I_threshold = I_threshold # Condition for done (the threshold for the number of infections)
        self.lockdown_threshold = lockdown_threshold # Condition for done (the threshold for lockdown)

        self.D_reward_func = D_reward_func # Episode end (failure) reward
        self.C_reward_func = C_reward_func # Episode end (succeed) reward
        self.period_model = build_period_model(period, betas_m, betas_s, gammas, thetas) # Epidemic transition model
        self.simulation_round = simulation_round # Do we round the simulation results
        self.mobility_decay = mobility_decay # Used to calculate accumalted mobility score
        self.reward_clip = reward_clip # Do we clip the reward

        self.init_ac_m = np.zeros(shape=(self.nb_regions,)) # Initialize the accumulated mobility loss in the first day.
        self.OD_mean_out = self.OD.mean(0).sum(-1) 
        
        print('Mobility decay',mobility_decay)

        if self.C_reward_func is not None:
            print('Set success reward')
        
        if self.D_reward_func is not None:
            print('Set failure reward')

        self.fixed_no_policy_days = fixed_no_policy_days
        self.fixed_no_policy_i = fixed_no_policy_i
        self.expert_dif = expert_dif
        self.r0 = []
        self.shuffle_OD = shuffle_OD

        # Noise (Not used in KDD DURLECA)
        self.od_obs_p = od_obs_p
        self.od_obs_noise = od_obs_noise
        self.od_misopt_p = od_misopt_p
        self.od_misopt_noise = od_misopt_noise
        self.state_import_p = state_import_p
        self.state_import_noise = state_import_noise
        self.state_obs_p = state_obs_p
        self.state_obs_noise = state_obs_noise
        self.state_obs_round = state_obs_round
        self.noises = [self.od_obs_p, self.od_obs_noise, self.od_misopt_p, self.od_misopt_noise, self.state_import_p, self.state_import_noise, self.state_obs_p, self.state_obs_noise, self.state_obs_round]
        
        # np.random.seed(seed)
        pass

    def set_no_noise(self):
        self.od_obs_p, self.od_obs_noise, self.od_misopt_p, self.od_misopt_noise, self.state_import_p, self.state_import_noise, self.state_obs_p, self.state_obs_noise = [0]*8
        self.state_obs_round = False
    
    def set_noise(self):
        self.od_obs_p, self.od_obs_noise, self.od_misopt_p, self.od_misopt_noise, self.state_import_p, self.state_import_noise, self.state_obs_p, self.state_obs_noise, self.state_obs_round = self.noises
        if self.state_import_noise > 0 and self.state_import_p > 0 :
            print('Use state import noise')
        if self.state_obs_noise > 0 and self.state_obs_p > 0 :
            print('Use state obs noise')
        if self.od_misopt_p > 0 and self.od_misopt_noise > 0:
            print('Use od misopt noise')

    def get_R0(self, last_state, current_state):
        # calculate estimate R0
        # R0 = beta / gamma

        c_sum = current_state[:,:4].sum(0)
        l_sum = last_state[:,:4].sum(0)

        beta = self.population_sum * (l_sum[0] - c_sum[0]) / (l_sum[1] * l_sum[0])
        gamma = (c_sum[2] - l_sum[2] + c_sum[3] - l_sum[3]) / l_sum[1]

        r0 = beta / gamma
        self.r0.append(r0)
        return r0
    
        
    def step(self, action, compute_expert_reward = True):
        # State transition

        SIRH = self.state[:,:4]
        SIRH_v = self.state[:,4:8]
        time = self.state[:,8:9]
        accumulated_m = self.state[:,9:10]
        accumulated_d = self.state[:,10:]
        start_time = self.time_step
        
        OD_d = self.OD[self.time_step:self.time_step+self.period]
        # OD obs noise. We have move this to main.py
        # if self.od_obs_p > 0 and self.od_obs_noise > 0:
        #     print('Use od obs noise')
        #     od_obs_flag = np.bitwise_and(np.random.rand(self.period,self.nb_regions,self.nb_regions) < self.od_obs_p, OD_d > 0) 
        #     OD_d_n = np.int32(OD_d * od_obs_flag * np.random.randn(self.period,self.nb_regions,self.nb_regions) * self.od_obs_noise)
        #     OD_d = np.int32(OD_d) + OD_d_n
        #     OD_d[OD_d<0] = 0 

        # Get OD
        if  type(action) == int:
            # const policy
            OD_time = OD_d * action
        elif action.shape[-1] == 104329:
            # edge mode
            action = action.reshape((self.nb_regions, self.nb_regions))
            OD_time = OD_d * action[np.newaxis,:]
        elif len(action) == 1 and action.shape[-1] == 1:
            # graph mode
            OD_time = OD_d * action[0]
        elif action.shape[-1] == self.nb_regions:
            # node mode
            action = action.reshape(self.nb_regions, 1)
            OD_time = OD_d * action[np.newaxis,:]
        
        # OD misopt noise
        if self.od_misopt_p > 0 and self.od_misopt_noise > 0:
            od_misopt_flag = np.random.rand(self.period,self.nb_regions,self.nb_regions) < self.od_misopt_p # 注意我这里要求只有OD >= 0都可以
            OD_time += OD_d * od_misopt_flag * np.random.randn(self.period,self.nb_regions,self.nb_regions) * self.od_misopt_noise # 注意这里是误报百分比
            OD_time[OD_time<0] = 0
   
        self.OD_ratio = OD_time.sum() / OD_d.sum()

        
        # Get next SIRH state
        SIRH_ = self.period_model.predict([SIRH[np.newaxis,:], OD_time[np.newaxis,:]])[0] 
        
        # Exp Dif (Not used)
        reward_expert = None
        if self.expert_dif is not None and compute_expert_reward == True:
            action_expert = self.expert_dif([self.state]).reshape(self.nb_regions,self.nb_regions)
            OD_expert = OD_d * action_expert[np.newaxis,:]
            SIRH_expert = self.period_model.predict([SIRH[np.newaxis,:], OD_expert[np.newaxis,:]])[0]
            accumulated_m_expert = self.mobility_decay * accumulated_m + (OD_d - OD_expert).sum(axis = (0,2))[:,np.newaxis]
            reward_expert = self.reward_func(SIRH_expert, OD_d, OD_expert, accumulated_m_expert, accumulated_d)

        # Accumulate Mobility
        accumulated_m = self.mobility_decay * accumulated_m + (OD_d - OD_time).sum(axis = (0,2))[:,np.newaxis]
        accumulated_d = self.OD_mean_out[:,np.newaxis]
        
        # Round
        if self.simulation_round == True:
            if (self.accumulated_time + self.time_step) // 24 >= 50:
                if np.isnan(SIRH_[:,1]).any() == True:
                    print('nan', self.time_step)
                    exit()
                SIRH_[:,1] = round_func(SIRH_[:,1])  
                SIRH_[:,3] = round_func(SIRH_[:,3]) 

        # Time step
        self.time_step += self.period
        time[:self.period] += self.period
        if self.time_step >= len(self.OD):
            self.accumulated_time += self.time_step
            self.time_step -= len(self.OD)
            time[:self.period] -= len(self.OD)
            if self.shuffle_OD == True:
                random_id = np.arange(31)
                np.random.shuffle(random_id)
                self.OD = self.OD.reshape(31,24,self.nb_regions,self.nb_regions)[random_id].reshape(744,self.nb_regions,self.nb_regions)


        # Import Noise, i.e., the noise of simulation (due to the wrong estimation of disease parameters)
        if self.state_import_noise > 0 and self.state_import_p > 0 :
            import_noise_flag = np.random.rand(self.nb_regions,4) < self.state_import_p
            SIRH_ += SIRH_ * import_noise_flag * np.random.randn(self.nb_regions,4) * self.state_import_noise 
            SIRH_[SIRH_<0] = 0

        # Next True State
        R_delta = SIRH_[:,2]-SIRH[:,2]
        H_delta = SIRH_[:,3]-SIRH[:,3]
        SI = SIRH_[:,0]+SIRH_[:,1]
        SI_delta = SI - SIRH_v[:,0]
        self.state = np.hstack([SIRH_, SI[:,np.newaxis], R_delta[:,np.newaxis], H_delta[:,np.newaxis], SI_delta[:,np.newaxis], time, accumulated_m, accumulated_d])

        # Next Observable state
        SIRH_obs_ = np.copy(SIRH_)
        if self.state_obs_noise > 0 and self.state_obs_p > 0 :
            obs_noise_flag = np.random.rand(self.nb_regions, 4) < self.state_obs_p
            SIRH_obs_ += SIRH_obs_ * obs_noise_flag * np.random.randn(self.nb_regions, 4) * self.state_obs_noise # 注意这里是误报百分比
            SIRH_obs_[SIRH_obs_<0] = 0

        if self.state_obs_round == True:
            SIRH_obs_ = np.round(SIRH_obs_)
        
        SIRH_obs = self.obs_state[:,:4] # SIRH_obs in the last time step
        R_delta_obs = SIRH_obs_[:,2]-SIRH_obs[:,2]
        H_delta_obs = SIRH_obs_[:,3]-SIRH_obs[:,3]
        SI_obs = SIRH_obs_[:,0]+SIRH_obs_[:,1]
        SI_delta_obs = SI - self.obs_state[:,4]
        self.obs_state = np.hstack([SIRH_obs_, SI_obs[:,np.newaxis], R_delta_obs[:,np.newaxis], H_delta_obs[:,np.newaxis], SI_delta_obs[:,np.newaxis], time, accumulated_m, accumulated_d])

        # Reward and Done
        done = False
        reward = None
        if self.accumulated_time >= self.total_time:
            done = True
            # 撑过总时间，默认之后感染清零
            if self.C_reward_func is not None:
                reward = self.reward_clip[1] * self.C_reward_func(np.inf) 

        elif self.I_threshold > 0 and SIRH_[:,1].mean() > self.I_threshold:
            # Infection exceeds the threshold
            print('-'*30,'Infection end','-'*30)
            done = True
            if self.D_reward_func is not None:
                reward = self.reward_clip[0] * self.D_reward_func(self.total_time - self.accumulated_time - self.time_step)        
        elif self.lockdown_threshold > 0 and (accumulated_m / (accumulated_d + 1e-7)).mean() > self.lockdown_threshold:
            # Lockdown exceeds the threshold
            print('-'*30,'Lockdown end','-'*30)
            done = True
            if self.D_reward_func is not None:
                reward = self.reward_clip[0] * self.D_reward_func(self.total_time - self.accumulated_time - self.time_step)
        elif self.C_reward_func is not None and SIRH_[:,1].max() <= 0.1: 
            # No more infections
            print('-'*30,'Success end','-'*30)
            done = True
            reward = self.reward_clip[1] * self.C_reward_func(self.total_time - self.accumulated_time - self.time_step)

        if reward is None:
            reward = self.reward_func(SIRH_, OD_d, OD_time, accumulated_m, accumulated_d)
            reward = max(reward, self.reward_clip[0])
            reward = min(reward, self.reward_clip[1])
            if self.expert_dif is not None and reward_expert is not None:
                reward_expert = max(reward_expert, self.reward_clip[0])
                reward_expert = min(reward_expert, self.reward_clip[1])
                reward = reward - reward_expert

        return self.obs_state, reward, done, {}
    
    def reset(self, location_id = None, infected_people = None):

        self.OD_ratio = 1
        
        # Initialize infections
        if location_id is None:
            location_id = np.random.randint(low=0, high = self.nb_regions)
            while self.population[location_id] == 0:
                location_id = np.random.randint(low=0, high = self.nb_regions)
            
        if infected_people is None:
            infected_people = np.random.randint(low = 10, high = 20)
            infected_people = min(infected_people, self.population[location_id])

        SIRH = np.zeros((self.nb_regions,4), np.uint16)
        SIRH[:,0] = self.population

        SIRH[location_id, 0] -= infected_people
        SIRH[location_id, 1] += infected_people

        SIRH_v = np.zeros((self.nb_regions,4), np.uint16)
        SIRH_v[:,0] = SIRH[:,0]+SIRH[:,1]

        self.time_step = 0
        self.accumulated_time = 0
        time = np.zeros((self.nb_regions,1))
        time[:self.period,0] = np.arange(self.period)

        accumulated_m = self.init_ac_m
        accumulated_d = self.OD_mean_out

        self.state = np.hstack([SIRH, SIRH_v, time, accumulated_m[:,np.newaxis], accumulated_d[:,np.newaxis]])
        self.obs_state = np.copy(self.state)

        if self.fixed_no_policy_i > 0:
            # This initialization is not used.
            action = 1
            while self.state[:,1].mean() < self.fixed_no_policy_i:
                _, _, _, _ = self.step(action, compute_expert_reward = False)

        if self.fixed_no_policy_days > 0:
            action = 1
            for i in range(self.fixed_no_policy_days * 24 // self.period):
                _, _, _, _ = self.step(action, compute_expert_reward = False)

        return self.obs_state
    
    def counts(self):
        return self.state[:,:4].sum(axis=0)
