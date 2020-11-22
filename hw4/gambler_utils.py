import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import gym
import gambler_env

import warnings
warnings.filterwarnings('ignore')


FIG_PATH = 'figures/'

RANDOM_SEED = 17
ENV_NAME = "gambler"

"""
PLOTTING FUNCTIONS
"""

def plot_deltas_rewards(deltas, rewards, name=""):
    if name != "":
        fig, ax = plt.subplots(1, 2, figsize=(18,5))
        ax[0].set_title(f"Delta V {name.replace('_', ' ')}")
        ax[0].set_ylabel('delta')
        ax[0].set_xlabel('iteration')
        ax[0].plot(np.arange(len(deltas)), deltas)
        ax[1].set_title(f"Value {name.replace('_', ' ')}")
        ax[1].set_ylabel('value')
        ax[1].set_xlabel('iteration')
        ax[1].plot(np.arange(len(rewards)), rewards)
        plt.savefig(FIG_PATH+f"{name}_delta_reward")
        plt.show()


def plot_action_value(env, policy, V, name=""):
    if name != "":
        fig, ax1 = plt.subplots(figsize=(15,8))

        color = 'tab:red'
        ax1.set_title(f"Action & Value plot {name.replace('_', ' ')}")
        ax1.set_xlabel('State')
        ax1.set_ylabel('Best action', color=color)
        ax1.plot(policy, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Value', color=color)  # we already handled the x-label with ax1
        ax2.plot(V, color=color)
        ax2.tick_params(axis='y', labelcolor=color)


        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        plt.savefig(FIG_PATH+f"{name}_action_value")
        plt.show()


"""
VALUE ITERATION
"""
# Inspired by https://github.com/dennybritz/reinforcement-learning/tree/master/DP

def one_step_lookahead(env, discount_factor, state, v):
    A = np.zeros(env.nA)
    for action in range(min(env.nA, state)):
        for prob, next_state, reward, done in env.P(state, action):
            A[action] += prob * (reward + discount_factor * v[next_state])
    return A


def value_iteration_step(env, discount_factor=0.995, max_iterations=100, threshold_delta=1e-4):
    V = np.zeros(env.nS)
    deltas = []
    rewards = []
    running_time_list = [0]
    start_time = time.time()

    for iteration in range(max_iterations):
        delta = 0.
        reward = 0.
        for state in range(env.nS):
            # Save previous state value
            previous_V = V[state]

            A = one_step_lookahead(env, discount_factor, state, V)

            # Value Iteration: update state value using the Bellman optimality equation
            best_action_value = np.max(A)
            V[state] = best_action_value

            # Update the delta of convergence to the maximum delta found so far
            delta = max(delta, np.abs(best_action_value - previous_V))

            # Best action
            best_action = np.argmax(V)

            reward += best_action_value

        deltas.append(delta)
        rewards.append(reward)
        
        running_time = time.time() - start_time
        running_time_list.append(running_time)
        
        if delta < threshold_delta:
            break;

    return (V, deltas, rewards, running_time_list)


def optimal_policy_function(env, V, discount_factor=0.995):
    # Create a deterministic policy
    optimal_policy  = np.zeros((env.nS, env.nA))
    optimal_policy_actions  = np.zeros(env.nS).astype(int)
    for state in range(env.nS):
        A = one_step_lookahead(env, discount_factor, state, V)
        best_action = np.argmax(A)

        # Deterministic best action
        optimal_policy[state, best_action] = 1.0
        optimal_policy_actions[state] = best_action

    return (optimal_policy, optimal_policy_actions)


def value_iteration_function(env, discount_factor=0.995, max_iterations=100, plot_grid=True, name=""):
    V, deltas, rewards, running_time_list = value_iteration_step(env, discount_factor=discount_factor, 
                                                                 max_iterations=max_iterations)
    optimal_policy, optimal_policy_actions = optimal_policy_function(env, V, discount_factor=discount_factor)

    plot_deltas_rewards(deltas, rewards, name)
    plot_action_value(env, optimal_policy_actions, V, name)

    stats = {}
    best_iteration = np.argmax(rewards)
    stats['max_iteration'] = best_iteration
    stats['max_reward'] = rewards[best_iteration]
    stats['running_time'] = running_time_list[best_iteration]
    stats['mean_reward'] = np.mean(rewards)

    return (V, optimal_policy, stats)



"""
POLICY ITERATION
"""

def policy_iteration_step(env, discount_factor, V, policy, max_iterations=100, start_time=0, threshold_delta=1e-4):
    deltas = []
    rewards = []
    running_time_list = []

    for iteration in range(max_iterations):
        delta = 0.
        reward = 0.
        for state in range(env.nS):
            # Save previous state value
            previous_V = V[state]

            # Update V
            Q_current_state = one_step_lookahead(env, discount_factor, state, V)
            V[state] = 0.
            for action in range(min(env.nA, state)):
                V[state] += policy[state, action] * Q_current_state[action]

            # Update the delta of convergence to the maximum delta found so far
            delta = max(delta, np.abs(V[state] - previous_V))

            reward += np.max(Q_current_state)

        deltas.append(delta)
        rewards.append(reward)

        running_time = time.time() - start_time
        running_time_list.append(running_time)

        if delta < threshold_delta:
            break;

    return (V, policy, deltas, rewards, running_time_list)


def policy_improvement(env, discount_factor, V, policy, policy_actions):
    continue_improve = False

    for state in range(env.nS):
        # New best action
        previous_best_action = np.argmax(policy[state])
        Q_current_state = one_step_lookahead(env, discount_factor, state, V)
        new_best_action = np.argmax(Q_current_state.round(decimals=4))

        # If policy has improved
        if previous_best_action != new_best_action:
            continue_improve = True

        # New policies
        policy[state] = np.eye(env.nA)[new_best_action]
        policy_actions[state] = new_best_action

    return (continue_improve, policy, policy_actions)


def policy_iteration_function(env, discount_factor=0.995, max_iterations_hl=10, max_iterations_ll=100, plot_grid=True, name=""):
    iteration = 0
    continue_improve = True
    optimal_V = np.zeros(env.nS)
    optimal_policy = np.ones([env.nS, env.nA]) / env.nA # Uniform action sampling initialization
    optimal_policy_actions = np.zeros(env.nS).astype(int)

    final_deltas = []
    final_rewards = []
    final_running_time = []
    start_time = time.time()

    while iteration <= max_iterations_hl and continue_improve:
        optimal_V, optimal_policy, deltas, rewards, running_time_list = policy_iteration_step(env, discount_factor=discount_factor, 
                                                                                              V=optimal_V, policy=optimal_policy, 
                                                                                              max_iterations=max_iterations_ll, 
                                                                                              start_time=start_time)

        continue_improve, optimal_policy, optimal_policy_actions = policy_improvement(env, discount_factor=discount_factor, 
                                                                                      V=optimal_V, 
                                                                                      policy=optimal_policy, 
                                                                                      policy_actions=optimal_policy_actions)
        final_deltas += deltas
        final_rewards += rewards
        final_running_time += running_time_list
        iteration += 1


    plot_deltas_rewards(final_deltas, final_rewards, name)
    plot_action_value(env, optimal_policy_actions, optimal_V, name)

    stats = {}
    best_iteration = np.argmax(final_rewards)
    stats['max_iteration'] = best_iteration
    stats['max_reward'] = final_rewards[best_iteration]
    stats['running_time'] = final_running_time[best_iteration]
    stats['mean_reward'] = np.mean(rewards)

    return (optimal_V, optimal_policy, stats)


"""
Q-LEARNING
"""

def epsilon_greedy_policy(env, Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(min(env.nA, state))
    else:
        return np.argmax(Q[state])


def plot_average_score_parameters(average_scores, liste, episode_max_score, name="", offset=20):
    if name != "":
        array_plot = np.array(liste)

        fig, ax = plt.subplots(1, 2, figsize=(15,5))
        ax[0].set_title(f"Epsilon/Learning rate progression {name.replace('_', ' ')}")
        ax[0].set_xlabel('episodes')
        ax[0].plot(array_plot[:,0], label="epsilon")
        ax[0].plot(array_plot[:,1], label="learning rate")
        ax[0].legend()
        ax[1].set_title(f"Reward progression {name.replace('_', ' ')}")
        ax[1].set_ylabel('average reward')
        ax[1].set_xlabel('episodes')
        ax[1].plot(average_scores)
        ax[1].axvline(x=episode_max_score, color="red", linestyle="--")
        plt.savefig(FIG_PATH+f"{name}_reward_parameters_progression")
        plt.show()


def q_learning_function(env, num_episodes=10000, discount_factor=0.995, initial_Q=None,
                        learning_rate_decay=0.999, epsilon_decay=0.999, 
                        plot_grid=True, name="", do_plot=True):

    offset = int(num_episodes/10)
    np.random.seed(RANDOM_SEED)
    start = time.time()
    env.reset()

    scores, average_scores = [], []
    running_time = 0
    episode = 0
    epsilon = 1
    learning_rate = 1
    list_epsilon_lr = [[epsilon, learning_rate]]

    Q = initial_Q if initial_Q is not None else np.zeros((env.nS, env.nA))

    max_score = -np.inf
    best_Q = np.zeros((env.nS, env.nA))

    # Training
    while True:
        score = 0.
        start_time = time.time()
        episode += 1
        state = env.reset()

        # Episode
        while True:
            action = epsilon_greedy_policy(env, Q, state, epsilon)
            new_state, reward, done, _ = env.step(action)

            Q[state, action] =  Q[state, action] * (1 - learning_rate) + \
                                     learning_rate * (reward + discount_factor * np.max(Q[new_state, :]))
            state = new_state
            score += reward

            if done:
                break

        scores.append(score)
        current_score = np.mean(scores[-500:])
        average_scores.append(current_score)

        running_time += time.time() - start_time

        if max_score < current_score and episode > offset:
            best_Q = Q
            max_score = current_score
            episode_max_score = episode
            running_time_max_score = round(running_time, 2)

        if episode % int(num_episodes/5) == 0 and do_plot:
            print('\nEpisode {}'.format(episode))
            print('average score = {:.4f}'.format(average_scores[-1]))

        if episode > num_episodes:
            if do_plot:
                print('\n---> finished at episode {} in {:.2f} seconds'.format(episode-1, time.time() - start))
                print(f"max score: {max_score} at episode {episode_max_score} after {running_time_max_score}s")
            break

        # Decay
        epsilon = max(0.01, epsilon_decay * epsilon)
        learning_rate = max(0.01, learning_rate_decay * learning_rate)

        list_epsilon_lr.append([epsilon, learning_rate])

    # Get final policy with the best Q
    optimal_V = np.zeros(env.nS)
    optimal_policy = np.zeros((env.nS, env.nA))
    optimal_policy_actions = np.zeros(env.nS).astype(int)

    for state in range(env.nS):
        best_action = np.argmax(best_Q[state])
        best_value = np.max(best_Q[state])

        optimal_policy[state, best_action] = 1.0
        optimal_V[state] = best_value
        optimal_policy_actions[state] = best_action

    # False positive
    for idx in range(len(average_scores)):
        if average_scores[idx] > max_score:
            average_scores[idx] = 0
    # Plotting
    plot_average_score_parameters(average_scores, list_epsilon_lr, episode_max_score, name, offset)
    plot_action_value(env, optimal_policy_actions, optimal_V, name)

    env.close()

    stats = {}
    stats['max_iteration'] = episode_max_score
    stats['max_reward'] = max_score
    stats['running_time'] = running_time_max_score
    stats['mean_reward'] = np.mean(average_scores)

    return (optimal_V, best_Q, stats)


"""
COMPARISON FUNCTIONS
"""

def play_function(env, V, max_episodes=100, max_iterations=100, do_print=False):
    list_scores = []
    for iteration in range(max_iterations):
        score = 0
        for _ in range(max_episodes):
            state = env.reset()
            #print("state", state)
            for t in range(10000):
                action = epsilon_greedy_policy(env, V, state, epsilon=0)
                state, reward, done, _ = env.step(action)
                #print("new state", state)
                #print("reward", reward)
                if done:
                    score += reward
                    break
        if do_print:
            print("Agent succeeded to reach goal {} out of {} Episodes using this policy for iteration {}".format(score, 
                                                                                                                  max_episodes, 
                                                                                                                  iteration))
        list_scores.append(score)

    mean_score = round(np.mean(list_scores)/max_episodes*100, 2)
    std_score = round(np.std(list_scores)/max_episodes*100, 2)

    stats = {}
    stats['mean_score'] = mean_score
    stats['std_score'] = std_score

    if do_print:
        print(f"\nFor {max_episodes} episodes:")
        print(f"Success rate: {mean_score}%")
        print("Standard deviation score:", std_score)
        print()
    env.close()
    
    return stats


