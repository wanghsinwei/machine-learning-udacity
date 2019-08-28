import numpy as np
from tasks.physics_sim import PhysicsSim
from agents.ou_noise import OUNoise
import pandas as pd
import random

# current_pos = np.array([1., 1., 1.])
# target_pos = np.array([10., 10., 10.])

# dist = np.linalg.norm(current_pos - target_pos)
# print(dist)

runtime = 5
init_pose = np.array([0., 0., 0., 0., 0., 0.])    # initial pose
init_velocities = np.array([0., 0., 0.])         # initial velocities
init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
target_pos = np.array([-3., 5., 10.])

# dones = np.array([True, False, True])
# print((1 - dones))

sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
last_dist = np.linalg.norm(target_pos - init_pose[:3])

# action = np.array([0, 0, 0, 0])
# action = action + np.array([-100, 0, 1, 904.5])
# print(action)
# print(np.clip(action, 0, 900))

last_pose = sim.pose[:3]
for i in range(200):
    # action = np.random.uniform(400, 900, 4)
    action = np.array([890, 900, 900, 900])
    # sim.reset()
    done = sim.next_timestep(action) # update the sim pose and velocities
    print('done: ' + str(done))
    target_vec = target_pos - sim.pose[:3]
    dist = np.linalg.norm(target_vec)
    target_unit_vec = target_vec / dist
    v_proj_target = np.inner(sim.v, target_unit_vec) * target_unit_vec

    print('\n' + str(i) + '  ===============')
    print('pos: ', end='')
    print(sim.pose[:3])
    print('EAngles: ', end='')
    print(sim.pose[3:])
    print('v:   ', end='')
    print(sim.v)
    # print('target_unit_vec: ', end='')
    # print(target_unit_vec)
    # print('v_proj_target => ', end='')
    # print(v_proj_target)
    # print('angv: ', end='')
    # print(sim.angular_v)
    print('dist: ' + str(dist))
    dist_impr = last_dist - dist
    print('last_dist - dist => ', end='')
    print(dist_impr)
    print('pose - last_pose => ', end='')
    print(sim.pose[:3] - last_pose)

    last_dist = dist
    last_pose = sim.pose[:3]

# exploration_mu = 0
# exploration_theta = 0.15
# exploration_sigma = 0.2
# noise = OUNoise(4, exploration_mu, exploration_theta, exploration_sigma)

# for i in range(100):
#     print(noise.sample())
#     # print(np.random.randn(4))
#     # print(random.randrange(420, 450))
#     # print(np.random.uniform(420, 450, 4))


# v = np.array([3,3,3])
# print(np.linalg.norm(v))
# target_vec = np.array([1,1,0])
# dist = np.linalg.norm(target_vec)
# target_unit_vec = target_vec / dist
# v_proj_target = np.inner(v, target_unit_vec) * target_unit_vec
# print(v_proj_target)

# print(np.sign(-5) == np.sign(-1))