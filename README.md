# FM_planning

## How to get dataset:
1. replace diffuser/diffuser/datasets/buffer.py with the buffer.py file in this repo.
2. download process_d4rl_data.py and put it in the same parent directory of the diffuser repo:
   
     Parent_Directory
   
           |
   
           | - - diffuser
   
           |
   
           |-- process_d4rl_data.py
 4. run process_d4rl_data.py
 5. the dataset will be stored in a processed_data folder within the same directory as process_d4rl_data.py


## How to access the data
1. The dataset contains different sections including observations, actions, rewards, next_observations, terminals, path_lengths.
2. Shape of observations: (3213, 1000, 11), which means there are 3213 trajectories and each trajectory has 1000 transitions, and the dimension of each observation is 11.
3. actions: (3213,1000,3); rewards: (3213,1000,1) ....
4. actions and observations are normalized into [-1,1]
5. load dataset:
   <pre> ''' python
   import numpy as np
   data = np.load('processed_data/hopper-medium-expert-v2_processed.npz')
   </pre>
7.<pre>
   observations = data['observations']     
   actions = data['actions']               
   rewards = data['rewards']              
   path_lengths = data['path_lengths']
   ...
   </pre>
9. Normalization parameters:
<pre>
   obs_mins = data['normalizer_mins_observations']
   obs_maxs = data['normalizer_maxs_observations']
   act_mins = data['normalizer_mins_actions']
   act_maxs = data['normalizer_maxs_actions']
</pre>
