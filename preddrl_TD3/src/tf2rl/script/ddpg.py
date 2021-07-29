import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, concatenate, Conv1D, Flatten, Dropout

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.misc.huber_loss import huber_loss


class Actor(tf.keras.Model):
    def __init__(self, state_shape, action_dim, max_action, units=[256, 256], name="Actor"):
        super().__init__(name=name)

        self.l1 = Dense(256, name="L1")#400 256
        self.l2 = Dense(256, name="L2")#300 256
        self.l3 = Dense(action_dim, name="L3")
        self.l4 = Dense(128, name="L4")
        self.l5 = Dense(256, name="L5")
        self.l6 = Conv1D(4,2,padding='same',strides=1,activation='relu',input_shape=(21,1),name="Conv1")
        self.l7 = Conv1D(4,2,padding='same',strides=1,activation='relu',name="Conv2")
        self.l8 = Conv1D(8,6,padding='same',strides=1,activation='relu',name="Conv3")
        self.l9 = Dense(128, name="L9")
        

        self.l10 = Dense(256, name="L10")
        self.l11 = Dense(256, name="L11")
        self.l12 = Dropout(rate=0.2)
         
        self.max_action = max_action

        with tf.device("/gpu:0"):
            self(tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float32)))

    def call(self, inputs):
        #DNN
        features = tf.nn.relu(self.l1(inputs))
        features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        action = self.max_action * tf.nn.tanh(features)
        return action

        #CNN-1 
        # input_ = tf.expand_dims(inputs,axis=-1)
        
        # features = self.l6(input_[:,:20])
        # # print(features)
        # # features = self.l7(features)
        # # print(features)
        # features = Flatten()(features)
        
        # features1 = tf.nn.relu(self.l1(features))

        # state = tf.concat([features1,inputs[:,20:24]],axis=-1)
        # # print(state)
        # features =  tf.nn.relu(self.l2(state))
        # features =  tf.nn.relu(self.l8(features))
        # features = self.l3(features)

        # CNN-2
        # input_ = tf.expand_dims(inputs,axis=-1)
        # # 3 Conv layers
        # features = self.l6(input_[:,:21])
        # # features = self.l7(features)
        # # features = self.l8(features)
        # print(features.shape) 
        # features = Flatten()(features)   
            
        # features1 = tf.nn.relu(self.l1(features))

        # # DNN layers
        # vel = tf.nn.relu(self.l10(inputs[:,21:23]))
        # polor = tf.nn.relu(self.l11(inputs[:,23:25]))
        # state = tf.concat([features1,vel,polor],axis=-1) #256x3
        
        # features = tf.nn.relu(self.l2(state)) #256
        # features = tf.nn.relu(self.l9(features))#128
        # # features = self.l12(features)
        # features = self.l3(features)
    
        # action = self.max_action * tf.nn.tanh(features)
        # return action
#
        

class Critic(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=[256, 256], name="Critic"):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1")
        self.l2 = Dense(units[1], name="L2")
        self.l3 = Dense(1, name="L3")

        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=[1, action_dim], dtype=np.float32))
        with tf.device("/gpu:0"):
            self([dummy_state, dummy_action])

    def call(self, inputs):
        states, actions = inputs
        features = tf.concat([states, actions], axis=1)
        features = tf.nn.relu(self.l1(features))
        features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        return features


class DDPG(OffPolicyAgent):
    def __init__(
            self,
            state_shape,
            action_dim,
            name="DDPG",
            max_action=1.,
            lr_actor=0.001,
            lr_critic=0.001,
            actor_units=[400, 300],
            critic_units=[400, 300],
            sigma=0.1,
            tau=0.005,
            n_warmup=int(1e4),
            memory_capacity=int(1e6),
            **kwargs):
        super().__init__(name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)

        # Define and initialize Actor network
        self.actor = Actor(state_shape, action_dim, max_action, actor_units)
        self.actor_target = Actor(
            state_shape, action_dim, max_action, actor_units)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        update_target_variables(self.actor_target.weights,
                                self.actor.weights, tau=1.)

        # Define and initialize Critic network
        self.critic = Critic(state_shape, action_dim, critic_units)
        self.critic_target = Critic(state_shape, action_dim, critic_units)
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_critic)
        update_target_variables(
            self.critic_target.weights, self.critic.weights, tau=1.)

        # Set hyperparameters
        self.sigma = sigma
        self.tau = tau

    def get_action(self, state, test=False, tensor=False):
        is_single_state = len(state.shape) == 1
        if not tensor:
            assert isinstance(state, np.ndarray)
        state = np.expand_dims(state, axis=0).astype(
            np.float32) if is_single_state else state
        action = self._get_action_body(
            tf.constant(state), self.sigma * (1. - test),
            tf.constant(self.actor.max_action, dtype=tf.float32))
        if tensor:
            return action
        else:
            return action.numpy()[0] if is_single_state else action.numpy()

    @tf.function
    def _get_action_body(self, state, sigma, max_action):
        with tf.device(self.device):
            action = self.actor(state)
            action += tf.random.normal(shape=action.shape,
                                       mean=0., stddev=sigma, dtype=tf.float32)
            return tf.clip_by_value(action, -max_action, max_action)

    def train(self, states, actions, next_states, rewards, done, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)
        actor_loss, critic_loss, td_errors = self._train_body(
            states, actions, next_states, rewards, done, weights)

        if actor_loss is not None:
            tf.summary.scalar(name=self.policy_name+"/actor_loss",
                              data=actor_loss)
        tf.summary.scalar(name=self.policy_name+"/critic_loss",
                          data=critic_loss)

        return td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, done, weights):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_errors = self._compute_td_error_body(
                    states, actions, next_states, rewards, done)
                critic_loss = tf.reduce_mean(
                    huber_loss(td_errors, delta=self.max_grad) * weights)

            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))

            with tf.GradientTape() as tape:
                next_action = self.actor(states)
                actor_loss = -tf.reduce_mean(self.critic([states, next_action]))

            actor_grad = tape.gradient(
                actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

            # Update target networks
            update_target_variables(
                self.critic_target.weights, self.critic.weights, self.tau)
            update_target_variables(
                self.actor_target.weights, self.actor.weights, self.tau)

            return actor_loss, critic_loss, td_errors

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        if isinstance(actions, tf.Tensor):
            rewards = tf.expand_dims(rewards, axis=1)
            dones = tf.expand_dims(dones, 1)
        td_errors = self._compute_td_error_body(
            states, actions, next_states, rewards, dones)
        return np.abs(np.ravel(td_errors.numpy()))

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        with tf.device(self.device):
            not_dones = 1. - dones
            target_Q = self.critic_target(
                [next_states, self.actor_target(next_states)])
            target_Q = rewards + (not_dones * self.discount * target_Q)
            target_Q = tf.stop_gradient(target_Q)
            current_Q = self.critic([states, actions])
            td_errors = target_Q - current_Q
        return td_errors
