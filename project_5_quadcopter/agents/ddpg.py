import numpy as np
from .replay_buffer import ReplayBuffer
from .ou_noise import OUNoise
from keras import layers, models, optimizers, regularizers
from keras import backend as K

class Actor:
    """Deep Deterministic Policy Gradients or DDPG: Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=400, activation='relu')(states)
        # net = layers.Dense(units=400)(states)
        # net = layers.BatchNormalization()(net)
        # net = layers.Activation('relu')(net)

        net = layers.Dense(units=300, activation='relu')(net)
        # net = layers.Dense(units=300)(states)
        # net = layers.BatchNormalization()(net)
        # net = layers.Activation('relu')(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        # Note that the raw actions produced by the output layer are in a [0, 1] range because of using a sigmoid activation function
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        # These gradients will need to be computed using the critic model, and fed in while training.
        # Hence it is specified as part of the "inputs" used in the below training function
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=0.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

class Critic:
    """Deep Deterministic Policy Gradients or DDPG: Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        # These two layers can first be processed via separate "pathways" (mini sub-networks), but eventually need to be combined.
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=400, activation='relu', kernel_regularizer=regularizers.l2(0.01))(states)
        # net_states = layers.Dense(units=400, kernel_regularizer=regularizers.l2(0.01))(states)
        # net_states = layers.BatchNormalization()(net_states)
        # net_states = layers.Activation('relu')(net_states)

        net_states = layers.Dense(units=300, activation='relu', kernel_regularizer=regularizers.l2(0.01))(net_states)
        # net_states = layers.Dense(units=300, kernel_regularizer=regularizers.l2(0.01))(net_states)
        # net_states = layers.BatchNormalization()(net_states)
        # net_states = layers.Activation('relu')(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=400, activation='relu', kernel_regularizer=regularizers.l2(0.01))(actions)
        net_actions = layers.Dense(units=300, activation='relu', kernel_regularizer=regularizers.l2(0.01))(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed
        net = layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(net)

        # Add final output layer to prduce action values (Q values)
        # The final output of this model is the Q-value for any given (state, action) pair.
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        # Compute the gradient of this Q-value with respect to the corresponding action vector, needed for training the actor model.
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

class DDPG:
    """Reinforcement Learning agent that learns using DDPG (Deep Deterministic Policy Gradients)."""
    def __init__(self, task, batch_size=64, gamma=0.99):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Put together the Actor and Critic models to build our DDPG agent. 
        # Note that we will need two copies of each model - one local and one target.
        # This is an extension of the "Fixed Q Targets" technique from Deep Q-Learning, and is used to decouple the parameters being updated from the ones that are producing target values.

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        self.last_noise_sample = None

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = gamma  # discount factor
        self.tau = 0.1  # for soft update of target parameters

    def reset_episode(self):
        self.noise.reset()
        self.last_noise_sample = None
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state, add_noise=True, noise_scale=1):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]

        if add_noise:
            self.last_noise_sample = self.noise.sample() * noise_scale
            action_with_noise = np.clip(action + self.last_noise_sample, self.action_low, self.action_high)
            return list(action_with_noise)  # add some noise for exploration
        else:
            return list(action)

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        # Notice that after training over a batch of experiences, we could just copy our newly learned weights (from the local model) to the target model.
        # However, individual batches can introduce a lot of variance into the process, so it's better to perform a soft update, controlled by the parameter tau.
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)