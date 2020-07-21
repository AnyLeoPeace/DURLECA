import tensorflow as tf

import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer, Dense, Dropout,Activation, Flatten, Input, Concatenate, Reshape, Lambda, Multiply, Permute, LSTM, RepeatVector
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from keras_radam import RAdam

from ddpg import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from model_layers import *

from exp_policy import get_exp_policy
import numpy as np
from param_noise import AdaptiveParamNoiseSpec


'''Build Model'''
def cross_concate(inputs):
    a,b = inputs
    a_ = tf.reshape(tf.tile(a, [1, 1, b.shape[1]]), (-1, a.shape[1] * b.shape[1], a.shape[2]))
    b_ = tf.tile(b, [1, a.shape[1], 1])
    return tf.concat([a_, b_], -1)

def get_sample_weight_func(args):
    # Not used
    sample_k = args.sample_k
    sample_b = args.sample_b

    def get_sample_weight(times):
        # Used to calculate sample weights according to time
        # The early the time step, the bigger the sample weights
        # We may let this function trainble in the future
        times_ = times - args.repeat * 744
        weights = 1 / (sample_b + np.exp(times_ * sample_k))

        return weights
    
    if args.use_sample_weights:
        return get_sample_weight
    
    else:
        return None

def get_nb_actions(action_mode, nb_regions):
    if action_mode == 'edge':
        nb_actions = nb_regions * nb_regions
    elif action_mode == 'node':
        nb_actions = nb_regions
    elif action_mode == 'graph':
        nb_actions = 1
    else:
        print('Wrong action mode')
        exit()
    return nb_actions

def get_build_func(OD_tensor, args):
    dim = args.dim
    delta = tf.constant(1e-7, dtype = tf.float32)

    nb_regions = K.int_shape(OD_tensor)[-1]
    nb_actions = get_nb_actions(args.action_mode, nb_regions)
    mobility_decay = tf.constant(args.mobility_decay, dtype=tf.float32)
    OD_mean = Lambda(lambda x: tf.reduce_mean(x, axis=0, keepdims = True))(OD_tensor) # 1 * 323 * nb_regions
    OD_mean_out = Lambda(lambda x: tf.reduce_mean(x, axis=-1))(OD_mean) # 1 * nb_regions



    def get_accumulated_cost(inputs):
        OD, OD_, accumlated = inputs
        return accumlated * mobility_decay + K.sum(OD - OD_, axis = [1,-1])

    def get_feature_input():
        print('Layer type', args.layer_type)
        if args.layer_type == 'weights':
            print('No visible Round')
            layer = Lambda(lambda x: tf.concat(
                (tf.math.reduce_sum(x[:,:,:4], axis=-1,keepdims = True), x[:,:,2:8]), axis=-1
                ))
        else:
            layer = Lambda(lambda x: x[:,:,2:8])

        return layer
    
    aggregate_op = None
    if args.layer_type == 'weights':
        print('Use OD Layer')
        GNN_layer = GraphSageConvOD
    elif args.layer_type == 'softmax':
        print('Use Softmax Layer')
        GNN_layer = GraphSageConvSoftmax
    else:
        print('Use',args.layer_type)
        aggregate_op = args.layer_type
        GNN_layer = GraphSageConv


    def build():
        state_input = Input(shape=(nb_regions, 11), name = 'state_input') 
        control_index = np.arange(0, args.period).reshape(4,-1)        

        def build_actor():
            print('-'*30+'Build Actor'+'-'*30)
            
            time_input = Lambda(lambda x: x[:,:args.period,8])(state_input) # None * period
            OD_input = Lambda(lambda x: tf.gather(OD_tensor, tf.cast(x,tf.int32), axis=0))(time_input)

            ac_m_input = Lambda(lambda x: K.expand_dims(K.repeat(x[:,:,9], nb_regions), axis = -1))(state_input) # nb_regions * nb_regions * 1
            ac_d_input = Lambda(lambda x: K.expand_dims(K.repeat(x[:,:,10], nb_regions), axis = -1))(state_input) # nb_regions * nb_regions * 1
            features_input = get_feature_input()(state_input)

            features = GNN_layer(16, aggregate_op = aggregate_op, BN = BatchNormalization(), activation='relu', kernel_regularizer=l2(5e-4), index = control_index[0])([features_input, OD_input])
            features = GNN_layer(32, aggregate_op = aggregate_op, BN = BatchNormalization(), activation='relu', kernel_regularizer=l2(5e-4), index = control_index[1])([features, OD_input])
            features = GNN_layer(dim, aggregate_op = aggregate_op, BN = BatchNormalization(), activation='relu', kernel_regularizer=l2(5e-4), index = control_index[2])([features, OD_input])
            features = GNN_layer(dim, aggregate_op = aggregate_op, BN = BatchNormalization(), activation='relu', kernel_regularizer=l2(5e-4), index = control_index[3])([features, OD_input])

            print('Action mode',args.action_mode)

            if args.action_mode == 'edge':
                OD_sum = Lambda(lambda x: K.sum(x,1))(OD_input) # nb_regions*nb_regions
                features = Lambda(cross_concate)([features, features]) # nb_regions*nb_regions, 2*dim
                OD_ = Permute((2,3,1))(OD_input) # nb_regions*nb_regions*4

                if args.layer_type == 'weights':
                    features = Reshape((nb_regions, nb_regions, dim*2+2))(features)
                else:
                    features = Reshape((nb_regions, nb_regions, dim*2))(features)

                features = Lambda(lambda inputs: tf.concat((inputs[0], inputs[1], inputs[2], inputs[3]), -1))([ac_m_input, ac_d_input, OD_, features])
                features = Dense(dim, activation = 'relu')(features)
                features = BatchNormalization()(features) # None * nb_regions * nb_regions * dim
                actions_ = Dense(1, activation='sigmoid')(features)
                actions_ = Reshape((nb_regions,nb_regions))(actions_)
                actions = Lambda(lambda x:  tf.where(tf.not_equal(x[1], 0), 
                                                    x[0], 
                                                    tf.ones_like(x[0]))
                                                    )([actions_, OD_sum])
                actions = Reshape((nb_regions*nb_regions,))(actions)

            elif args.action_mode == 'node':
                # nb_regions actions
                OD_sum = Lambda(lambda x: tf.reduce_sum(x,axis=[1,-1]))(OD_input) # nb_regions*nb_regions
                features = Lambda(lambda inputs: tf.concat((inputs[0][:,0], inputs[1][:,0], tf.expand_dims(inputs[2], -1), inputs[3]), -1))([ac_m_input, ac_d_input, OD_sum, features])
                features = Dense(dim, activation = 'relu')(features)
                features = BatchNormalization()(features)
                actions = Dense(1, activation='sigmoid')(features)
                actions = Reshape((nb_regions,))(actions)

            elif args.action_mode == 'graph':
                # 1 action
                OD_sum = Lambda(lambda x: tf.reduce_sum(x,axis=[1,-1]))(OD_input) # nb_regions*nb_regions
                features = Lambda(lambda inputs: tf.concat((inputs[0][:,0], inputs[1][:,0], tf.expand_dims(inputs[2], -1), inputs[3]), -1))([ac_m_input, ac_d_input, OD_sum, features])
                features = Flatten()(features)
                features = Dense(dim*8, activation = 'relu')(features)
                features = BatchNormalization()(features)
                actions = Dense(1, activation='sigmoid')(features)

            else:
                print('Wrong action mode')
                exit()

            actor = Model(state_input, actions)

            return actor

        def build_critic():
            print('-'*30+'Build Critic'+'-'*30)

            action_input = Input(shape=(nb_actions,), name='critic/action_input')
            ac_m_input = Lambda(lambda x: x[:,:,9])(state_input)
            ac_d_input = Lambda(lambda x: x[:,:,10])(state_input)

            time_input = Lambda(lambda x: x[:,:args.period,8])(state_input)
            OD_input = Lambda(lambda x: tf.gather(OD_tensor, tf.cast(x,tf.int32), axis=0))(time_input)
            features_input = get_feature_input()(state_input)
            
            print('Action mode',args.action_mode)
            if args.action_mode == 'edge':
                # nb_regions*nb_regions actions
                action_ = Reshape((nb_regions, nb_regions))(action_input)
            elif args.action_mode == 'node':
                # nb_regions actions
                action_ = Lambda(lambda x: tf.expand_dims(x,-1))(action_input)
            elif args.action_mode == 'graph':
                # one action
                action_ = action_input
            else:
                print('Wrong action mode')
                exit()

            OD_ = Multiply()([action_, OD_input])
            ac_m = Lambda(get_accumulated_cost)([OD_input, OD_, ac_m_input])
            ac_d = ac_d_input
            
            if 'avg' in args.reward_func:
                print('Avg mode')
                current_ratio = Lambda(lambda x: tf.math.divide_no_nan(tf.reduce_sum(x[0],axis=[1,-1]), delta+x[1]))([OD_, ac_d])
            else:
                print('Norm mode')
                current_ratio = Lambda(lambda x: tf.math.divide_no_nan(tf.reduce_sum(x[0],axis=[1,-1]), delta+tf.reduce_sum(x[1], axis=[1,-1])))([OD_, OD_input])

            ac_ratio = Lambda(lambda x: tf.math.divide_no_nan(x[0], delta+x[1]))([ac_m, ac_d])

            features = GNN_layer(16, aggregate_op = aggregate_op, BN = BatchNormalization(), activation='relu', kernel_regularizer=l2(5e-4), index = control_index[0])([features_input, OD_])
            features = GNN_layer(32, aggregate_op = aggregate_op, BN = BatchNormalization(), activation='relu', kernel_regularizer=l2(5e-4), index = control_index[1])([features, OD_])
            features = GNN_layer(dim, aggregate_op = aggregate_op, BN = BatchNormalization(), activation='relu', kernel_regularizer=l2(5e-4), index = control_index[2])([features, OD_])
            features = GNN_layer(dim, aggregate_op = aggregate_op, BN = BatchNormalization(), activation='relu', kernel_regularizer=l2(5e-4), index = control_index[3])([features, OD_])

            if args.pool_type == 'flatten':
                print('Graph Pool Type: Flatten')
                reward = Flatten()(features)
                reward = Lambda(lambda x: tf.concat((x[0],x[1],x[2],x[3]), axis=-1))([ac_ratio, current_ratio, ac_d, reward])
                reward = Dense(dim*8, activation = 'relu')(reward)
                reward = BatchNormalization()(reward)
                reward = Dense(dim*2, activation = 'relu')(reward)
                reward = BatchNormalization()(reward)

            elif args.pool_type == 'weight_sum':
                print('Graph Pool Type: WeightSum')
                OD_mean_out_ = Lambda(lambda x: tf.gather(OD_mean_out, tf.cast(x[:,args.period,8],tf.int32), axis=0))(state_input) 
                features = Lambda(lambda x: tf.concat((tf.expand_dims(x[0],-1),tf.expand_dims(x[1],-1),tf.expand_dims(x[2],-1),x[3]), axis=-1))([ac_ratio, current_ratio, ac_d, features])
                features = Dense(dim, activation = 'relu')(features)
                features = BatchNormalization()(features)
                reward = GraphSoftmaxPool()([features, OD_mean_out_])   

            elif args.pool_type == 'weight_dense':
                print('Graph Pool Type: WeightDenseSum')
                OD_mean_out_ = Lambda(lambda x: tf.gather(OD_mean_out, tf.cast(x[:,args.period,8],tf.int32), axis=0))(state_input) # None * nb_regions 
                features = Lambda(lambda x: tf.concat((tf.expand_dims(x[0],-1),tf.expand_dims(x[1],-1),tf.expand_dims(x[2],-1),x[3]), axis=-1))([ac_ratio, current_ratio, ac_d, features])
                features = Dense(dim, activation = 'relu')(features)
                features = BatchNormalization()(features)
                edge = Lambda(lambda x: tf.expand_dims(x, -1))(OD_mean_out_)
                edge = Dense(16, activation='relu')(edge)
                edge = BatchNormalization()(edge)
                edge = Dense(1)(edge)
                edge = Flatten()(edge)
                reward = GraphSoftmaxPool()([features, edge])   

            else:
                print('Not Supported Pool Type')

            reward = Dense(1)(reward)

            critic = Model(inputs=[action_input, state_input], outputs=reward)

            return critic
        
        return build_actor, build_critic
    
    return build

def build_agent(OD, OD_tensor, args):
    '''Build model'''
    build_func = get_build_func(OD_tensor, args)
    nb_regions = OD.shape[-1]
    nb_actions = get_nb_actions(args.action_mode, nb_regions)

    def get_prob_imitation(steps):
        if steps < args.prob_imitation_steps:
            p = (1 - 1 / (1 + np.exp((-steps / args.prob_imitation_steps + 0.5)*10))) * args.base_prob_imitation
        else:
            p = 0
        
        return max(p, args.min_prob_imitation)
    
    def get_std_adapt():
        if args.std_adapt_steps <= 0:
            return None

        def std_adapt(steps):
            if steps < args.std_adapt_steps:
                return 1 - 1 / (1 + np.exp((-steps / args.std_adapt_steps + 0.5)*10))
            else:
                return 0
        
        return std_adapt

    memory = SequentialMemory(limit=args.memory_limit, window_length=1)

    if args.action_noise == True:
        random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.rd_theta, mu=0, sigma=args.rd_sigma, dt=args.rd_dt)
    else:
        random_process = None

    if args.param_noise == True:
        param_noise = AdaptiveParamNoiseSpec(initial_stddev = args.init_std, desired_action_stddev = args.action_std, adoption_coefficient = args.adapt, min_action_std=args.min_action_std,std_adapt=get_std_adapt())
    else:
        param_noise = None
    

    agent = DDPGAgent(nb_actions=nb_actions, build_func=build_func, nb_regions=nb_regions,
                    start_step=args.start_step,
                    memory=memory, nb_steps_warmup_critic=args.warmup_steps, nb_steps_warmup_actor=args.warmup_steps, 
                    exp_policy = get_exp_policy(OD, args),
                    batch_size = args.batch_size, param_noise = param_noise, get_prob_imitation = get_prob_imitation,
                    train_interval=args.train_interval, 
                    random_process=random_process, gamma=args.decay, target_model_update=args.update, delta_clip=args.delta_clip)

    agent.compile(eval(args.optimizer)(lr=args.lr, clipnorm=1.), metrics = ['mae'])

    return agent

