import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import base64
from pathlib import Path
from IPython import display as ipythondisplay


def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
  """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """
  eval_env = DummyVecEnv([lambda: gym.make(env_id)])
  # Start the video at step=0 and record 500 steps
  eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix=prefix)

  obs = eval_env.reset()
  for _ in range(video_length):
    action, _ = model.predict(obs)
    obs, _, _, _ = eval_env.step(action)

  # Close the video recorder
  eval_env.close()


def show_videos(video_path='', prefix=''):
  """
  Taken from https://github.com/eleurent/highway-env

  :param video_path: (str) Path to the folder containing videos
  :param prefix: (str) Filter the video, showing only the only starting with this prefix
  """
  html = []
  for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
      video_b64 = base64.b64encode(mp4.read_bytes())
      html.append('''<video alt="{}" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>'''.format(mp4, video_b64.decode('ascii')))
  ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


model = PPO('MlpPolicy', "CartPole-v1", verbose=1).learn(10000)

record_video('CartPole-v1', model, video_length=500, prefix='ppo2-cartpole')
show_videos('videos', prefix='ppo2')

#
# env = gym.make('CartPole-v1')
#
# model = PPO(MlpPolicy, env, verbose=0)
#
#
# def evaluate(model, num_episodes=100):
#     """
#     Evaluate a RL agent
#     :param model: (BaseRLModel object) the RL Agent
#     :param num_episodes: (int) number of episodes to evaluate it
#     :return: (float) Mean reward for the last num_episodes
#     """
#     # This function will only work for a single Environment
#     env = model.get_env()
#     all_episode_rewards = []
#     for i in range(num_episodes):
#         episode_rewards = []
#         done = False
#         obs = env.reset()
#         while not done:
#             # _states are only useful when using LSTM policies
#             action, _states = model.predict(obs)
#             # here, action, rewards and dones are arrays
#             # because we are using vectorized env
#             obs, reward, done, info = env.step(action)
#             episode_rewards.append(reward)
#
#         all_episode_rewards.append(sum(episode_rewards))
#
#     mean_episode_reward = np.mean(all_episode_rewards)
#     print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)
#
#     return mean_episode_reward
#
#
# # Random Agent, before training
# mean_reward_before_train = evaluate(model, num_episodes=100)
#
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
#
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
#
# # Train the agent for 10000 steps
# model.learn(total_timesteps=10000)
# # Evaluate the trained agent
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
#
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")