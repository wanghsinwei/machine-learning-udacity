import numpy as np
from .physics_sim import PhysicsSim

class ReachTargetTask:
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 12 # 3 Euler angles + 3 velocity + 3 angle velocity + target vector
        self.action_low = 1 # Use a value larger than 0 to avoid the error of divide by zero
        self.action_high = 900
        self.action_size = 4 # 4 rotors

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.goal_dist = 0.5
        self.last_target_vec_abs = np.absolute(self.target_pos - self.sim.pose[:3])

        self.has_takeoff = False

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        target_vec = self.target_pos - self.sim.pose[:3]
        dist = np.linalg.norm(target_vec)
        # target_unit_vec = target_vec / dist
        # v_proj_target = np.inner(self.sim.v, target_unit_vec) * target_unit_vec
        # reward = 0.1*v_proj_target[2] + 0.01*(v_proj_target[0] + v_proj_target[1])

        current_target_vec_abs = np.absolute(target_vec)
        target_vec_abs_diff = self.last_target_vec_abs - current_target_vec_abs
        self.last_target_vec_abs = current_target_vec_abs

        # Distance Reward
        reward = 0.5*target_vec_abs_diff[2] if target_vec_abs_diff[2] > 0 else target_vec_abs_diff[2]
        reward += 0.5*target_vec_abs_diff[0] if target_vec_abs_diff[0] > 0 else target_vec_abs_diff[0]
        reward += 0.5*target_vec_abs_diff[1] if target_vec_abs_diff[1] > 0 else target_vec_abs_diff[1]

        # Velocity Reward
        reward += 0.02 if np.sign(target_vec[2]) == np.sign(self.sim.v[2]) else -0.04
        reward += 0.02 if np.sign(target_vec[0]) == np.sign(self.sim.v[0]) else -0.02
        reward += 0.02 if np.sign(target_vec[1]) == np.sign(self.sim.v[1]) else -0.02

        # Constant Reward for Keep Flying
        if self.sim.pose[2] > 0:
            reward += 0.1

        if dist <= self.goal_dist:
            reward = 20 # Reach Goal
        elif self.sim.done:
            if not self.has_takeoff:
                reward = -20 # No Takeoff Penalty
            elif self.sim.pose[2] <= 0:
                reward = -10 # Crash Penalty
            elif self.sim.time <= self.sim.runtime:
                reward = -5 # Cross Boundary Penalty

        return np.tanh(reward)

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        state_all = []
        for i in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            if not self.has_takeoff and self.sim.pose[2] > 0:
                self.has_takeoff = True
            target_vec = self.target_pos - self.sim.pose[:3]
            if not done and np.linalg.norm(target_vec) <= self.goal_dist: # Reach Goal
                done = True
            reward += self.get_reward() 
            state_all.append(self.sim.pose[3:])
            state_all.append(self.sim.v)
            state_all.append(self.sim.angular_v)
            state_all.append(target_vec)
        next_state = np.concatenate(state_all)

        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.has_takeoff = False
        target_vec = self.target_pos - self.sim.pose[:3]
        self.last_target_vec_abs = np.absolute(target_vec)
        state = np.concatenate([self.sim.pose[3:], self.sim.v, self.sim.angular_v, target_vec] * self.action_repeat)
        return state

    def sample_act(self):
        return np.random.uniform(self.action_low, self.action_high, self.action_size)