import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Flatten, concatenate

from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.misc.huber_loss import huber_loss

from ddpg import DDPG, Actor


class Critic(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=[400, 300], name="Critic"):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1")
        self.l2 = Dense(units[1], name="L2")
        self.l3 = Dense(1, name="L3")

        self.l4 = Dense(units[0], name="L4")
        self.l5 = Dense(units[1], name="L5")
        self.l6 = Dense(1, name="L6")

        # self.l7 = Conv1D(4,20,padding='same',strides=2,activation='relu',input_shape=(-1,20,1),name="Conv1")
        # self.l8 = Conv1D(4,20,padding='same',strides=2,activation='relu',input_shape=(-1,20,1),name="Conv2")   
        # self.l9 = Dense(64, name="L9")
        # self.l10 = Dense(64, name="L10")
        # self.l11 = Dense(64, name="L11")
        # self.l12 = Dense(64, name="L12")
        # self.l13 = Dense(64, name="L13")
        # self.l14 = Dense(64, name="L14")
         
        
        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=[1, action_dim], dtype=np.float32))
        with tf.device("/gpu:0"):
            self([dummy_state, dummy_action])
            

    def call(self, inputs):
        states, actions = inputs
        
        xu = tf.concat([states, actions], axis=1)
        # xu_ = tf.expand_dims(xu,axis=-1)
        #
        x1 = tf.nn.relu(self.l1(xu))
        x1 = tf.nn.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = tf.nn.relu(self.l4(xu))
        x2 = tf.nn.relu(self.l5(x2))
        x2 = self.l6(x2)
        #

        # x1 = self.l7(xu_[:,:20])
        # x1 = Flatten()(x1)
        # x1 = tf.nn.relu(self.l9(x1))#64

        # x1_2 = tf.nn.relu(self.l1(xu[:,20:22])) #64
        # x1_3 = tf.nn.relu(self.l2(xu[:,22:24])) #64

        # x1_4 = tf.nn.relu(self.l13(xu[:,24:26])) #64

        # x1_ = concatenate([x1, x1_2, x1_3, x1_4],axis=-1)
        # x1_1 = tf.nn.relu(self.l11(x1_))
        # x1_f = self.l3(x1_1)

        # x2 = self.l8(xu_[:,:20])
        # x2 = Flatten()(x2)
        # x2 = tf.nn.relu(self.l10(x2)) #64

        # x2_2 = tf.nn.relu(self.l4(xu[:,20:22])) #64
        # x2_3 = tf.nn.relu(self.l5(xu[:,22:24])) #64

        # x2_4 = tf.nn.relu(self.l14(xu[:,24:26])) #64

        # x2_ = concatenate([x2, x2_2, x2_3, x2_4],axis=-1)
        # x2_1 = tf.nn.relu(self.l12(x2_))
        # x2_f = self.l6(x2_1)

        # return x1_f, x2_f
        return x1, x2


class TD3(DDPG):
    def __init__(
            self,
            state_shape,
            action_dim,
            name="TD3",
            actor_update_freq=2,
            policy_noise=0.2,
            noise_clip=0.5,
            actor_units=[400, 300],
            critic_units=[400, 300],
            lr_critic=0.001,
            **kwargs):
        super().__init__(name=name, state_shape=state_shape, action_dim=action_dim,
                         actor_units=actor_units, critic_units=critic_units,
                         lr_critic=lr_critic, **kwargs)

        self.critic = Critic(state_shape, action_dim, critic_units)
        self.critic_target = Critic(state_shape, action_dim, critic_units)
        update_target_variables(
            self.critic_target.weights, self.critic.weights, tau=1.)
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_critic)

        self._policy_noise = policy_noise
        self._noise_clip = noise_clip

        self._actor_update_freq = actor_update_freq
        self._it = tf.Variable(0, dtype=tf.int32)

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, done, weights):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_error1, td_error2 = self._compute_td_error_body(
                    states, actions, next_states, rewards, done)
                critic_loss = tf.reduce_mean(huber_loss(td_error1, delta=self.max_grad) * weights) + \
                              tf.reduce_mean(huber_loss(td_error2, delta=self.max_grad) * weights)

            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))

            self._it.assign_add(1)
            with tf.GradientTape() as tape:
                next_actions = self.actor(states)
                actor_loss = - \
                    tf.reduce_mean(self.critic([states, next_actions]))
                # actor_loss = - \
                #     tf.reduce_mean(self.critic([states[:,:20],states[:,20:22],states[:,22:24], next_actions]))
            remainder = tf.math.mod(self._it, self._actor_update_freq)
            def optimize_actor():
                actor_grad = tape.gradient(
                    actor_loss, self.actor.trainable_variables)
                return self.actor_optimizer.apply_gradients(
                    zip(actor_grad, self.actor.trainable_variables))

            tf.cond(pred=tf.equal(remainder, 0), true_fn=optimize_actor, false_fn=tf.no_op)
            # Update target networks
            update_target_variables(
                self.critic_target.weights, self.critic.weights, self.tau)
            update_target_variables(
                self.actor_target.weights, self.actor.weights, self.tau)

            return actor_loss, critic_loss, tf.abs(td_error1) + tf.abs(td_error2)

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        td_errors1, td_errors2 = self._compute_td_error_body(
            states, actions, next_states, rewards, dones)
        return np.squeeze(np.abs(td_errors1.numpy()) + np.abs(td_errors2.numpy()))

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        with tf.device(self.device):
            not_dones = 1. - dones

            # Get noisy action
            next_action = self.actor_target(next_states)
            noise = tf.cast(tf.clip_by_value(
                tf.random.normal(shape=tf.shape(next_action),
                                 stddev=self._policy_noise),
                -self._noise_clip, self._noise_clip), tf.float32)
            next_action = tf.clip_by_value(
                next_action + noise, -self.actor_target.max_action, self.actor_target.max_action)

            target_Q1, target_Q2 = self.critic_target(
                [next_states, next_action])
            # target_Q1, target_Q2 = self.critic_target(
            #     [next_states[:,:20], next_states[:,20:22], next_states[:,22:24], next_action])
            
            target_Q = tf.minimum(target_Q1, target_Q2)
            target_Q = rewards + (not_dones * self.discount * target_Q)
            target_Q = tf.stop_gradient(target_Q)

            current_Q1, current_Q2 = self.critic([states, actions])
            # print(states.shape)
            # current_Q1, current_Q2 = self.critic([states[:,:20],states[:,20:22],states[:,22:24], actions])

        return target_Q - current_Q1, target_Q - current_Q2
