import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plotly.offline.init_notebook_mode(connected=True)

def plot_position(csv_file, size=None):
    """Visualize how the position and velocity of the quadcopter evolved during the simulation."""
    results = pd.read_csv(csv_file)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=size)
    fig.patch.set_facecolor('white')
    # ax1.set_title('Position')
    ax1.plot(results['time'], results['x'], label='x')
    ax1.plot(results['time'], results['y'], label='y')
    ax1.plot(results['time'], results['z'], label='z')
    ax1.legend()
    ax1.set_ylabel('Position')

    ax2.plot(results['time'], results['x_velocity'], label='x_hat')
    ax2.plot(results['time'], results['y_velocity'], label='y_hat')
    ax2.plot(results['time'], results['z_velocity'], label='z_hat')
    ax2.legend()
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Velocity')

def plot_euler_angle(csv_file, size=None):
    """Plot the Euler angles (the rotation of the quadcopter over the  𝑥 -,  𝑦 -, and  𝑧 -axes) and the velocities (in radians per second) corresponding to each of the Euler angles."""
    results = pd.read_csv(csv_file)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=size)
    fig.patch.set_facecolor('white')
    ax1.plot(results['time'], results['phi'], label='phi φ')
    ax1.plot(results['time'], results['theta'], label='theta θ')
    ax1.plot(results['time'], results['psi'], label='psi ψ')
    ax1.legend()
    ax1.set_ylabel('Euler Angle')
    
    ax2.plot(results['time'], results['phi_velocity'], label='phi_velocity')
    ax2.plot(results['time'], results['theta_velocity'], label='theta_velocity')
    ax2.plot(results['time'], results['psi_velocity'], label='psi_velocity')
    ax2.legend()
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Euler Angle Velocity')

def plot_rotor_speed(csv_file, size=None):
    """Plot the agent's choice of actions (rotor speed)."""
    results = pd.read_csv(csv_file)

    fig = plt.figure(figsize=size)
    fig.patch.set_facecolor('white')
    # plt.title('Rotor Speed')

    plt.plot(results['time'], results['rotor_speed1'], label='Rotor 1')
    plt.plot(results['time'], results['rotor_speed2'], label='Rotor 2')
    plt.plot(results['time'], results['rotor_speed3'], label='Rotor 3')
    plt.plot(results['time'], results['rotor_speed4'], label='Rotor 4')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Rotor Speed (revolutions / second)')

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

def plot_rewards(rewards_list, mean_reward_epi_num = 10):
    eps, rews = np.array(rewards_list).T
    smoothed_rews = running_mean(rews, mean_reward_epi_num)

    fig=plt.figure(figsize=(12, 6), facecolor='w', edgecolor='k')
    plt.plot(eps[-len(smoothed_rews):], smoothed_rews, label='Mean Reward')
    plt.plot(eps, rews, color='grey', alpha=0.5, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()

def plot_trajectory_3d(csv_file, title, filename, width=600, height=600, axis_ranges=([-1,1], [-1,1], [-1,1])):
    df = pd.read_csv(csv_file)

    trace = go.Scatter3d(
        x=df['x'], y=df['y'], z=df['z'],
        marker=dict(
            size=4,
            color=df['time'],
            colorscale='Portland'
        ),
        line=dict(
            color='#000000',
            width=1
        )
    )

    data = [trace]

    layout = dict(
        title=title,
        width=width,
        height=height,
        autosize=False,
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 0, 0)',
                showbackground=True,
                backgroundcolor='rgb(220, 220, 220)',
                range=axis_ranges[0]
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 0, 0)',
                showbackground=True,
                backgroundcolor='rgb(220, 220, 220)',
                range=axis_ranges[1],
                color='green'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 0, 0)',
                showbackground=True,
                backgroundcolor='rgb(180, 180, 180)',
                range=axis_ranges[2],
                color='blue'
            ),
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=2, y=2, z=0.1)
            )
        )
    )

    fig = dict(data=data, layout=layout)

    plotly.offline.iplot(fig, filename=filename, validate=True)