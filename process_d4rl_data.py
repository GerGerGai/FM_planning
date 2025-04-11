import os
import numpy as np
import gym
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import d4rl
import torch
from diffuser.datasets.preprocessing import get_preprocess_fn
from diffuser.datasets.normalization import DatasetNormalizer, LimitsNormalizer
from diffuser.datasets.buffer import ReplayBuffer
from diffuser.datasets.d4rl import sequence_dataset
import argparse

@contextmanager
def suppress_output():
    """Suppress output from D4RL loading"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

def load_environment(name):
    """Load D4RL environment"""
    if type(name) != str:
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def process_and_save_data(env_name, output_dir, horizon=64, max_path_length=1000, 
                         max_n_episodes=10000, termination_penalty=0, preprocess_fns=None):
    """
    Process D4RL dataset in the same format as diffuser and save to file
    
    Args:
        env_name: Name of D4RL environment (e.g., 'hopper-medium-expert-v2')
        output_dir: Directory to save processed data
        horizon: Sequence length for diffusion model
        max_path_length: Maximum length of a trajectory
        max_n_episodes: Maximum number of episodes
        termination_penalty: Penalty for terminal states
        preprocess_fns: List of preprocessing functions to apply
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load environment
    print(f"Loading environment: {env_name}")
    env = load_environment(env_name)
    
    # Get preprocessing function
    if preprocess_fns is None:
        # Use appropriate preprocessing based on environment
        if 'hopper' in env_name:
            preprocess_fns = []  # No special preprocessing needed for hopper
        elif 'maze2d' in env_name:
            preprocess_fns = ['maze2d_set_terminals']
        else:
            preprocess_fns = []
    
    preprocess_fn = get_preprocess_fn(preprocess_fns, env)
    
    # Load and process sequences
    print("Loading and processing sequences...")
    itr = sequence_dataset(env, preprocess_fn)
    
    # Create replay buffer
    fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
    for i, episode in enumerate(itr):
        fields.add_path(episode)
    fields.finalize()
    
    # Normalize data
    print("Normalizing data...")
    normalizer = DatasetNormalizer(fields, LimitsNormalizer, path_lengths=fields['path_lengths'])
    
    # Normalize observations and actions
    normalized_observations = normalizer.normalize(fields['observations'], 'observations')
    normalized_actions = normalizer.normalize(fields['actions'], 'actions')
    normalized_next_observations = normalizer.normalize(fields['next_observations'], 'observations')
    
    # Save processed data
    output_path = os.path.join(output_dir, f"{env_name}_processed.npz")
    print(f"Saving processed data to: {output_path}")
    
    # Save all fields
    np.savez(
        output_path,
        observations=normalized_observations,
        actions=normalized_actions,
        rewards=fields['rewards'],
        next_observations=normalized_next_observations,
        terminals=fields['terminals'],
        path_lengths=fields['path_lengths'],
        normalizer_mins_observations=normalizer.normalizers['observations'].mins,
        normalizer_maxs_observations=normalizer.normalizers['observations'].maxs,
        normalizer_mins_actions=normalizer.normalizers['actions'].mins,
        normalizer_maxs_actions=normalizer.normalizers['actions'].maxs,
        horizon=horizon,
        max_path_length=max_path_length,
        termination_penalty=termination_penalty
    )
    
    # Print dataset information
    print("\nProcessed Dataset Information:")
    print("----------------------------")
    print(f"Number of episodes: {fields.n_episodes}")
    print(f"Total steps: {fields.n_steps}")
    print(f"Observations shape: {normalized_observations.shape}")
    print(f"Actions shape: {normalized_actions.shape}")
    print(f"Path lengths: {fields['path_lengths'][:5]}...")  # Show first 5 path lengths
    
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper-medium-expert-v2',
                       help='D4RL environment name')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                       help='Directory to save processed data')
    parser.add_argument('--horizon', type=int, default=64,
                       help='Sequence length for diffusion model')
    parser.add_argument('--max_path_length', type=int, default=1000,
                       help='Maximum length of a trajectory')
    parser.add_argument('--max_n_episodes', type=int, default=10000,
                       help='Maximum number of episodes')
    parser.add_argument('--termination_penalty', type=float, default=0,
                       help='Penalty for terminal states')
    args = parser.parse_args()
    
    output_path = process_and_save_data(
        args.env, 
        args.output_dir,
        horizon=args.horizon,
        max_path_length=args.max_path_length,
        max_n_episodes=args.max_n_episodes,
        termination_penalty=args.termination_penalty
    )
    print(f"\nProcessing complete. Data saved to: {output_path}")

if __name__ == '__main__':
    main() 