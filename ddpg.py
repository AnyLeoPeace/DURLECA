'''The DDPG agent was orginallly from keras-rl https://github.com/keras-rl/keras-rl'''
'''We add parameter-noise and expert-imitation'''

from collections import deque
import os
import warnings

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import binary_accuracy
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input

from rl.core import Agent
from rl.random import OrnsteinUhlenbeckProcess
from rl_util import clone_optimizer, get_soft_target_model_updates, AdditionalUpdatesOptimizer, huber_loss
import tensorflow as tf
import warnings
from copy import deepcopy
from tensorflow.keras.callbacks import History
from tensorflow.keras.utils import multi_gpu_model

from rl.callbacks import (
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer
)


def mean_q(y_true, y_pred):
    return K.mean(y_pred)

# Deep DPG as described by Lillicrap et al. (2015)
# http://arxiv.org/pdf/1509.02971v2.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4324&rep=rep1&type=pdf
class DDPGAgent(Agent):
    def __init__(self, nb_actions, build_func, memory, start_step = 0, nb_regions = 323,
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000, 
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                 random_process=None, custom_model_objects={}, target_model_update=.001, 
                 exp_policy = None, param_noise = None, get_prob_imitation = lambda x: 0, 
                 **kwargs):

        super(DDPGAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        self.delta = tf.constant(1e-7, dtype = tf.float32) # Used for divide

        # Parameters.
        self.start_step = start_step
        self.nb_actions = nb_actions
        self.nb_steps_warmup_actor = nb_steps_warmup_actor + start_step
        self.nb_steps_warmup_critic = nb_steps_warmup_critic + start_step
        self.exp_policy = exp_policy # Smart policy func used to imitate
        self.get_prob_imitation = get_prob_imitation
        self.nb_regions = nb_regions

        # noise
        self.param_noise = param_noise
        self.random_process = random_process
        self.action_noise_flag = False

        self.delta_clip = delta_clip
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.build_func = build_func
        self.critic_action_input_idx = 0
        self.memory = memory

        # State.
        self.compiled = False
        self.reset_states()

        # Build
        self.build_models()
    

    @property
    def prob_imitation(self):
        return self.get_prob_imitation(self.step)

    
    def apply_param_noise(self, actor, perturbed_actor, param_noise):
        distance = 10
        self.action_noise_flag = False
        i = 0
        if self.memory.nb_entries <= self.batch_size or self.step < self.nb_steps_warmup_actor:
            perturbed_actor.set_weights(actor.get_weights())
        
        else:
            desired_action_stddev = param_noise.get_desired_action_stddev(self.step)
            while distance > desired_action_stddev*2 or distance < desired_action_stddev/2:
                # Apply param noise
                i = i + 1
                ws = actor.get_weights()
                for index, w in enumerate(ws):
                    ws[index] = w + np.random.normal(loc=0, scale=param_noise.current_stddev, size=w.shape)
                
                perturbed_actor.set_weights(ws)
                print('Set param-perturbed actor', i)

                # Distance
                experiences = self.memory.sample(self.batch_size)
                assert len(experiences) == self.batch_size

                # Start by extracting the necessary parameters (we use a vectorized implementation).
                state0_batch = []
                reward_batch = []
                action_batch = []
                terminal1_batch = []
                state1_batch = []
                for e in experiences:
                    state0_batch.append(e.state0[0])
                    state1_batch.append(e.state1[0]) # State1 is the delayed (acted) state after state0
                    reward_batch.append(e.reward)
                    action_batch.append(e.action)
                    terminal1_batch.append(0. if e.terminal1 else 1.)

                # Prepare and validate parameters.
                state0_batch = self.process_state_batch(state0_batch)
                state1_batch = self.process_state_batch(state1_batch)
                terminal1_batch = np.array(terminal1_batch)
                reward_batch = np.array(reward_batch)
                action_batch = np.array(action_batch)
                assert reward_batch.shape == (self.batch_size,)
                assert terminal1_batch.shape == reward_batch.shape
                assert action_batch.shape == (self.batch_size, self.nb_actions)            

                distance = self.distance_measure.predict_on_batch(state0_batch)
                if np.isnan(distance):
                    print('Nan distance, discard pertubation')
                    perturbed_actor.set_weights(actor.get_weights())
                    break
                else:
                    param_noise.adapt(distance, self.step)
            
                if i > 10:
                    if distance > desired_action_stddev*2:
                        print('Large distance, discard pertubation')
                        perturbed_actor.set_weights(actor.get_weights())
                        break
                    else:
                        break



    @property
    def uses_learning_phase(self):
        return True

    # Modification
    def build_models(self):
        build_actor, build_critic = self.build_func()
        self.actor = build_actor()
        self.critic = build_critic()

        target_build_actor, target_build_critic = self.build_func()
        self.target_actor = target_build_actor()
        self.target_critic = target_build_critic()

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        if self.param_noise is not None:
            self.perturbed_actor = build_actor()
            actor_input = self.actor.input
            OD_input = self.actor.layers[2].output
            OD_sum = Lambda(lambda x: tf.reshape(tf.cast(K.sum(x, axis = 1), tf.bool), (-1, self.nb_regions*self.nb_regions)))(OD_input)
            perturbed_actor_output = self.perturbed_actor(actor_input)
            if self.nb_actions == self.nb_regions * self.nb_regions:
                distance_func = lambda inputs: tf.sqrt(tf.reduce_sum(tf.math.divide_no_nan(
                        tf.square(inputs[0] - inputs[1]), self.delta + tf.reduce_sum(tf.cast(inputs[2], tf.float32))
                        )))
                distance = Lambda(distance_func)([self.actor.output, perturbed_actor_output, OD_sum])
            elif self.nb_actions == 1 or self.nb_actions == self.nb_regions:
                distance =  Lambda(lambda inputs: tf.sqrt(tf.reduce_mean(tf.square(inputs[0] - inputs[1]))))([self.actor.output, perturbed_actor_output])

            print('-'*20,'Build Perturbed Actor and Distance Measure','-'*20)
            self.distance_measure = Model(self.actor.input, distance)

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError('More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            loss = K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)
            return loss

        ## Compile target networks. We only use them in feed-forward mode, hence we can pass any
        ## optimizer and loss since we never use it anyway.
        self.target_actor.compile(optimizer='adam', loss='binary_crossentropy')            
        self.target_critic.compile(optimizer='adam', loss='binary_crossentropy')
        self.actor.compile(loss='binary_crossentropy', metrics = ['binary_accuracy'], optimizer = 'adam')

        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer('critic', critic_optimizer, critic_updates)
        
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=critic_metrics)

        # compile actor
        self.actor_optimizer = actor_optimizer
        actor_optimizer = self.actor_optimizer
        combined_inputs = [self.actor.outputs[0], self.actor.inputs[0]]
        combined_output = self.critic(combined_inputs)

        updates = actor_optimizer.get_updates(
            params=self.actor.trainable_weights, loss=-K.mean(combined_output) + self.actor.losses) # Actor is to maximize critic output
        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for BN
        self.actor_train_fn = K.function(self.actor.inputs,
                                            self.actor.outputs, updates=updates)
        
        # compile perturbed actor
        if self.param_noise is not None:
            print('Compile perturbed_actor')
            self.perturbed_actor.compile(optimizer='adam', loss='binary_crossentropy')
            self.perturbed_actor_optimizer = clone_optimizer(optimizer)

            combined_inputs = [self.perturbed_actor.outputs[0], self.perturbed_actor.inputs[0]]
            combined_output = self.critic(combined_inputs)

            updates = self.perturbed_actor_optimizer.get_updates(
            params=self.perturbed_actor.trainable_weights, loss=-K.mean(combined_output) + self.perturbed_actor.losses)

            updates += self.perturbed_actor.updates  # include other updates of the actor, e.g. for BN
            self.perturbed_actor_train_fn = K.function(self.perturbed_actor.inputs,
                                            self.perturbed_actor.outputs, updates=updates)
                  
        self.compiled = True
        

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def update_target_models_hard(self):
        self.target_critic.set_weights(self.critic.get_weights())
        self.target_actor.set_weights(self.actor.get_weights())

    # TODO: implement pickle

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
            self.action_noise_flag = False

        self.recent_action = None
        self.recent_observation = None

        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def select_target_action(self, state):
        if np.random.rand() < self.prob_imitation:
            # This is used for imitating expert.
            action = self.exp_policy(state)
        else:
            action = self.target_actor.predict_on_batch(state)
        
        return action

    def select_action(self, state):
        batch = self.process_state_batch(state)

        if not self.training:
            action = self.actor.predict_on_batch(batch)

        else:
            if np.random.rand() < self.prob_imitation:
                # This is used for imitation learing
                action = self.exp_policy(state)

            else:
                action = self.actor.predict_on_batch(batch)
                if self.param_noise is not None:
                    action = self.perturbed_actor.predict_on_batch(batch)
            
                if (self.random_process is not None) and (self.action_noise_flag == True):
                    noise = self.random_process.sample()
                    assert noise.shape == action.shape
                    action += noise

                    # Because our policy is between 0 and 1
                    action[action<0] = 0
                    action[action>1] = 1

        return action

    def forward(self, observation):
        # Select an action.
        state = observation[np.newaxis,:]
        action = self.select_action(state).flatten() # Here, add flatten, because forward is facing only one sample input

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = np.float16(action)
        self.next_state = state

        return action

    @property
    def layers(self):
        return self.actor.layers[:] + self.critic.layers[:]

    @property
    def metrics_names(self):
        names = self.critic.metrics_names[:]
        names += ['Actor_acc']

        if self.processor is not None:
            names += self.processor.metrics_names[:]

        return names

    def backward(self, reward, terminal=False):
        if abs(reward) > 10000:
            print('reward > 10000')
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                            training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
        if can_train_either and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0[0])
                state1_batch.append(e.state1[0]) # State1 is the delayed (acted) state after state0
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)
            

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions)    

            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:
                target_actions = self.select_target_action(state1_batch)
                # target_actions = self.target_actor.predict_on_batch(state1_batch)
                assert target_actions.shape == (self.batch_size, self.nb_actions)
                if len(self.critic.inputs) >= 3:
                    state1_batch_with_action = state1_batch[:]
                else:
                    state1_batch_with_action = [state1_batch]
                state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
                target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
                assert target_q_values.shape == (self.batch_size,)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.gamma * target_q_values
                discounted_reward_batch *= terminal1_batch
                assert discounted_reward_batch.shape == reward_batch.shape
                targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

                # Perform a single batch update on the critic network.
                if len(self.critic.inputs) >= 3:
                    state0_batch_with_action = state0_batch[:]
                else:
                    state0_batch_with_action = [state0_batch]

                state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
                metrics = self.critic.train_on_batch(state0_batch_with_action, targets)

                if self.processor is not None:
                    metrics += self.processor.metrics

            # Update actor
            actor_metrics = [np.nan]
            if self.step > self.nb_steps_warmup_actor:
                if len(self.actor.inputs) >= 2:
                    inputs = state0_batch[:]
                else:
                    inputs = [state0_batch]
            
                action_values = self.actor_train_fn(inputs)[0]

                if self.param_noise is not None:
                    # update perturbed actor
                    _ = self.perturbed_actor_train_fn(inputs)[0]

                if np.isnan(action_values).any():
                    action_values = action_values.reshape(-1, self.nb_regions, self.nb_regions)
                    pos_x, pos_y, pos_z = np.where(np.isnan(action_values))
                    l = len(pos_x)
                    print('Nan in train')
                    print('Critic metrics', metrics)
                    for index in range(l):
                        x = pos_x[index]
                        y = pos_y[index]
                        z = pos_z[index]
                        print(x,y,z)
                        print('Predicted action', action_values[x,y,z])
                        print('Time',state0_batch[x,:,8])
                        print('Infection', state0_batch[x,:,1])
                        print('Original action', action_batch[x,y,z])
                        print('Targets',targets[x,y,z])
                        print('Reward',reward_batch[x,y,z])
                        print('-'*30)
                        exit()
                
            metrics = metrics + actor_metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()

        return metrics


    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None, max_start_func = None):
        '''My modification: Add param noise'''
        
        """Trains the agent on the given environment.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose ==2:
            callbacks += [TrainEpisodeLogger()]
        elif verbose > 2:
            callbacks += [TrainIntervalLogger(interval=log_interval), TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        episode = np.uint32(0)
        self.step = np.uint32(0) + self.start_step
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                if observation is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = np.int16(0)
                    episode_reward = np.float32(0)

                    # apply noise to perturbed actor
                    if self.param_noise is not None:
                        self.apply_param_noise(self.actor, self.perturbed_actor, self.param_noise)

                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    assert observation is not None

                    # Perform random starts at beginning of episode and do not record them into the experience.
                    # This slightly changes the start position between games.
                    nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
                    
                    if max_start_func is None:
                        for _ in range(nb_random_start_steps):
                            if start_step_policy is None:
                                action = env.action_space.sample()
                            else:
                                action = start_step_policy(observation)
                            if self.processor is not None:
                                action = self.processor.process_action(action)
                            callbacks.on_action_begin(action)
                            observation, reward, done, info = env.step(action)
                            observation = deepcopy(observation)
                            if self.processor is not None:
                                observation, reward, done, info = self.processor.process_step(observation, reward, done, info)
                            callbacks.on_action_end(action)
                            if done:
                                warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
                                observation = deepcopy(env.reset())
                                if self.processor is not None:
                                    observation = self.processor.process_observation(observation)
                                break
                    else:
                        while max_start_func(observation) == False:
                            if start_step_policy is None:
                                action = env.action_space.sample()
                            else:
                                action = start_step_policy(observation)
                            if self.processor is not None:
                                action = self.processor.process_action(action)
                            callbacks.on_action_begin(action)
                            observation, reward, done, info = env.step(action)
                            observation = deepcopy(observation)
                            if self.processor is not None:
                                observation, reward, done, info = self.processor.process_step(observation, reward, done, info)
                            callbacks.on_action_end(action)
                            if done:
                                warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `fixed_no_policy_range` parameter.'.format(nb_random_start_steps))
                                observation = deepcopy(env.reset())
                                if self.processor is not None:
                                    observation = self.processor.process_observation(observation)
                                break


                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step.
                callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = np.float32(0)
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, done, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, done, info = self.processor.process_step(observation, r, done, info)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(action)
                    reward += r
                    if done:
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                metrics = self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': env.OD_ratio,
                    'observation': observation[:,1].mean(axis=0),
                    'reward': reward,
                    'metrics': metrics,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    self.forward(observation)
                    self.backward(0., terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None

        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        return history