import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf
from utils import simulation_round
import tensorflow.keras.backend as K

def get_next_tf_state_function(beta_m, beta_s, gamma, theta):
    delta = tf.constant(1e-7, dtype = tf.float32)

    def next_state_function(inputs):
        SIRH, OD = inputs
        SIRH = SIRH[0]
        SIR = SIRH[:,:3]
        OD = OD[0]

        # Hospitalized people would not move
        populations = K.sum(SIR, axis=1)
        SIR_n = tf.math.divide_no_nan(SIR, K.expand_dims(populations,-1) + delta)
        N = int(SIR.shape[0])

        # As the regional population is affected by previous mobility restrictions, the current move-out population may be more than the total population in very few cases.
        # Here we force the move-out population <= the total population
        ratio = tf.math.divide_no_nan(populations, K.sum(OD, axis=1) + delta)
        ratio = K.expand_dims(ratio,-1)
        ratio = K.repeat(ratio, N)[:,:,0]
        OD = tf.where(ratio < 1, OD*ratio,OD)
        
        OD_m = K.expand_dims(OD, axis = -1)
        OD_m_SIR = OD_m * K.repeat(SIR_n, N)

        inflow_healthy = K.sum(OD_m_SIR[:,:,0], axis=0)
        inflow_infected = K.sum(OD_m_SIR[:,:,1], axis=0)
        inflow_all = K.sum(K.sum(OD_m_SIR, axis=-1), axis=0)

        stay_healthy = SIR[:,0] - K.sum(OD_m_SIR[:,:,0], axis=1)
        stay_infected = SIR[:,1] - K.sum(OD_m_SIR[:,:,1],axis=1)
        stay_all = populations - K.sum(K.sum(OD_m_SIR,axis=-1),axis=1)
        
        # The "SIR^" in our paper.
        SIR = SIR - K.sum(OD_m_SIR, axis = 1) + K.sum(OD_m_SIR, axis = 0)

        # infected
        m_infected = tf.math.divide_no_nan(beta_m * inflow_healthy * inflow_infected, inflow_all + delta)
        s_infected = tf.math.divide_no_nan(beta_s * stay_healthy * stay_infected, stay_all + delta)
        new_infected = m_infected + s_infected
        new_infected = tf.where(new_infected > SIR[:, 0], SIR[:, 0], new_infected)

        # hospitaled
        new_hospitaled = gamma * SIR[:, 1]

        # recovered
        new_recovered = theta * SIRH[:, 3]   

        # Update SIR
        SIRH = K.stack([
            SIR[:, 0] - new_infected, 
            SIR[:, 1] + new_infected - new_hospitaled,
            SIR[:, 2] + new_recovered,
            SIRH[:, 3] + new_hospitaled - new_recovered
        ], axis = -1)
                
        return K.expand_dims(SIRH,0)
    
    return next_state_function


def build_state_model(beta_m, beta_s, gamma, theta):
    N = beta_m.shape[0]

    SIRH_input = Input(shape=(N, 4))
    OD_input = Input(shape=(N, N))

    next_state_function = get_next_tf_state_function(beta_m, beta_s, gamma, theta)
    SIR = Lambda(next_state_function)([SIRH_input, OD_input])
    state_model = Model([SIRH_input, OD_input], SIR)

    return state_model



def build_period_model(period, beta_m, beta_s, gamma, theta):
    N = beta_m.shape[0]

    features_input = Input(shape=(N, 4), name = 'period/features_input') # 3-channel (SIR)
    OD_input = Input(shape=(period, N, N), name = 'period/OD_input')
    state_model = build_state_model(beta_m, beta_s, gamma, theta)

    def period_func(inputs):
        features_input, OD_input = inputs
        sir = features_input
        for time in range(period):
            sir = state_model([sir, OD_input[:, time]])                
        
        return sir

    SIR = Lambda(period_func)([features_input, OD_input])
    period_model = Model([features_input, OD_input], SIR)

    return period_model
