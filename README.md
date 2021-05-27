[image1]: assets/reinforce_rep.png "image1"
[image2]: assets/eq_1.png "image2"
[image3]: assets/eq_2.png "image3"
[image4]: assets/eq_3.png "image4"
[image5]: assets/eq_4.png "image5"
[image6]: assets/eq_5.png "image6"
[image7]: assets/eq_6.png "image7"
[image8]: assets/8.png
[image9]: assets/9.png
[image10]: assets/10.png
[image11]: assets/11.png
[image12]: assets/12.png
[image13]: assets/13.png
[image14]: assets/14.png
[image15]: assets/15.png
[image16]: assets/16.png

# Deep Reinforcement Learning Theory - Proximal Policy Optimization

## Content 
- [Introduction](#intro)
- [REINFORCE](#reinforce)
- [Problems of REINFORCE](#problems)
- [Noise Reduction](#noise_red)
- [Rewards Normalization](#rewards_norm)
- [Credit Assignement](#credit)
- [Pong with REINFORCE](#pong)
- [Importance Sampling](#imp_samp)
- [Proximal Policy Optimization - PPO](#ppo)
- [PPO with Clipping](#ppo_clip)
- [Pong with PPO](#pong_ppo)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## Introduction <a name="what_is_reinforcement"></a>
- Reinforcement learning is **learning** what to do — **how to map situations to actions** — so as **to maximize a numerical reward** signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html))
- Deep reinforcement learning refers to approaches where the knowledge is represented with a deep neural network

### Overview:
- ***Policy-Based Methods***: methods such as 
    - hill climbing
    - simulated annealing
    - adaptive noise scaling
    - cross-entropy methods
    - evolution strategies

- ***Policy Gradient Methods***:
    - REINFORCE
    - lower the variance of policy gradient algorithms

- ***Proximal Policy Optimization***:
    - Proximal Policy Optimization (PPO), a cutting-edge policy gradient method

- ***Actor-Critic Methods***
    - how to combine value-based and policy-based methods
    - bringing together the best of both worlds, to solve challenging reinforcement learning problems

## REINFORCE <a name="reinforce"></a> 
- Brief review of the REINFORCE algorithm. 
    ![image1]

## Problems of REINFORCE <a name="problems"></a> 
- **Inefficient update process**! Run the policy once, update once, and then throw away the trajectory.
- The gradient estimate **g** is very **noisy**. 
- There is **no clear credit assignment**. A trajectory may contain many good/bad actions and whether these actions are reinforced depends only on the final total output.

## Noise Reduction <a name="noise_red"></a> 
- Optimize the policy by maximizing the average rewards **U(θ)**. 
- To do that: use **stochastic gradient ascent**. 
- Gradient is given by an **average over all the possible trajectories**:

    ![image2]

- **Millions of trajectories** for simple problems, and **infinite for continuous** problems possible.
- Take **one trajectory to compute the gradient**
- After training for a long time, the tiny signal accumulates.
- Using **distributed computing**, we can **collect multiple trajectories in parallel**, so that it won’t take too much time. 
- Estimate the **policy gradient by averaging** across all the different trajectories.

    ![image3]

## Rewards Normalization <a name="rewards_norm"></a> 
- There is another bonus for running multiple trajectories: we can collect all the total rewards and get a sense of how they are distributed.
- Learning can be improved by normalization of rewards, where **μ** the mean, and **σ** the standard deviation.

    ![image4]

## Credit Assignement <a name="credit"></a>
- Going back to the gradient estimate, we can take a closer look at the total reward **R**, which is just a sum of reward at each step **R = r<sub>1</sub> + r<sub>1</sub> +...+ r<sub>t-1</sub> + r<sub>t</sub> +...**
- At time-step **t**: Even before an action is decided, the agent has already received all the rewards up until step **t−1** (reward from the past). The rest is denoted as the future reward. 

    ![image5]

- Due to a Markov process: the action at time-step **t** can only affect the future reward, so the past reward shouldn’t be contributing to the policy gradient. 
- So ignore the past reward. A better policy gradient would simply have the future reward as the coefficient.
- It turns out that mathematically, ignoring past rewards might change the gradient for each specific trajectory, but it doesn't change the **averaged gradient**. 

    ![image6]

### Example
- Agent with two possible actions:
    - 0 = Do nothing
    - 1 = Move
- Three time-steps in each game
- Policy is completely determined by **θ**, such that the probability of "moving" is **θ**, and the probability of doing nothing is **1−θ**
- Initially **θ=0.5**. Three games are played, the results are:
    - Game 1: actions: (1,0,1) rewards: (1,0,1)
    - Game 2: actions: (1,0,0) rewards: (0,0,1)
    - Game 3: actions: (0,1,0) rewards: (1,0,1)

- What are the future rewards for the first game? 
    - **Result**: (1+0+1, 1+0, 1) = (2,1,1)
- What is the policy gradient computed from the second game, using future rewards?
    - **Result**: -2 
    - The future rewards are computed to be (1,1,1). 
    - Each time an action 1 is taken, it contributes **+1/0.5 = +2** to the policy gradient. This is because **π<sub>θ</sub>(1∣s) = θ**
    - Whereas everytime an action 0 is taken, it contributes **-1/0.5 = -2** to the gradient. This is because **π<sub>θ</sub>(0∣s) = 1−θ**
- These statements are true regarding the 3rd game
    - The contribution to the gradient from the second and third steps cancel each other
    - The computed policy gradient is -2
    - Total reward (2) vs. future rward give the same policy gradient


## Pong with REINFORCE <a name="pong"></a> 
- Open Jupyter Notebook ```pong_reinforce.ipynb```
- Try to teach an agent playing Pong by using only the pixels
- PongDeterministic-v4 does not contain random frameskipping --> less noise, easier to train
- Only **two actions** to be used: RIGHTFIRE, LEFTFIRE
- Output will provide two probabilities: 
    - Probability taking action to the right
    - Probability taking action to the left
- Preprocessed image: cropped to 80 x 80 pixels, black and white
- Two consecutive time frames are stacked together
- Final image: 80 x 80 pixels x 2 channels as input
    ### Install package for displaying animation
    ```
    # install package for displaying animation
    !pip install JSAnimation

    # custom utilies for displaying animation, collecting rollouts and more
    import pong_utils

    %matplotlib inline

    # check which device is being used. 
    # I recommend disabling gpu until you've made sure that the code runs
    device = pong_utils.device
    print("using device: ",device)
    ```
    ### Render AI gym environment 
    ```
    # render ai gym environment
    import gym
    import time

    # PongDeterministic does not contain random frameskip
    # so is faster to train than the vanilla Pong-v4 environment
    env = gym.make('PongDeterministic-v4')

    print("List of available actions: ", env.unwrapped.get_action_meanings())

    # we will only use the actions 'RIGHTFIRE' = 4 and 'LEFTFIRE" = 5
    # the 'FIRE' part ensures that the game starts again after losing a life
    # the actions are hard-coded in pong_utils.py

    RESULTS:
    ------------
    List of available actions:  ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
    ```
    ### Preprocessing
    ```
    import matplotlib
    import matplotlib.pyplot as plt

    # show what a preprocessed image looks like
    env.reset()
    _, _, _, _ = env.step(0)
    # get a frame after 20 steps
    for _ in range(20):
        frame, _, _, _ = env.step(1)

    plt.subplot(1,2,1)
    plt.imshow(frame)
    plt.title('original image')

    plt.subplot(1,2,2)
    plt.title('preprocessed image')

    # 80 x 80 black and white image
    plt.imshow(pong_utils.preprocess_single(frame), cmap='Greys')
    plt.show()
    ```
    ![image8]
    ### Implement a policy
    ```
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # set up a convolutional neural net
    # the output is the probability of moving right
    # P(left) = 1-P(right)
    class Policy(nn.Module):
        """ Define a policy based on a neural network
        """

        def __init__(self):
            """ Init of class Policy()
            
                INPUTS:
                ------------
                    None
                    
                OUTPUTS:
                ------------
                    No direct
            """
            
            super(Policy, self).__init__()
            
            # 80x80 to outputsize x outputsize
            # outputsize = (inputsize - kernel_size + stride)/stride 
            # (round up if not an integer)

            # conv1 80x80 --> 40x40 (due to stride=2)
            self.conv1 = nn.Conv2d(2, 4, kernel_size=2, stride=2)
            # conv2 40x40 --> 20x20 (due to stride=2)
            self.conv2 = nn.Conv2d(4, 8, kernel_size=2, stride=2)
            # conv3 20x20 --> 10x10 (due to stride=2)
            self.conv3 = nn.Conv2d(8, 16, kernel_size=2, stride=2)
            self.size=16*10*10
            
            # 3 fully connected layer
            self.fc1 = nn.Linear(self.size, 64)
            self.fc2 = nn.Linear(64, 8)
            self.fc3 = nn.Linear(8, 1)
            self.sig = nn.Sigmoid()
            
        def forward(self, x):
            """ Forward path of neural network
            
                INPUTS:
                ------------
                    x - (torch tensor) shape torch.Size([1, 2, 80, 80]) 
                        1 --> number of parallel instances, up to 4 at the moment
                        2 --> two consecutive stacked frames
                        80x80 --> pixel width and height
                    
                OUTPUTS:
                ------------
                    x - (torch tensor) shape torch.Size([1, 1]) - Probability to choose RIGHT as action
            """
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            
            # flatten the tensor
            x = x.view(-1,self.size)
            
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.sig(self.fc3(x))
            return x

    # use your own policy!
    policy=Policy().to(device)
    print(policy)

    #policy=pong_utils.Policy().to(device)

    # we use the adam optimizer with learning rate 2e-4
    # optim.SGD is also possible
    import torch.optim as optim
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    RESULTS:
    ------------
    Policy(
    (conv1): Conv2d(2, 4, kernel_size=(2, 2), stride=(2, 2))
    (conv2): Conv2d(4, 8, kernel_size=(2, 2), stride=(2, 2))
    (conv3): Conv2d(8, 16, kernel_size=(2, 2), stride=(2, 2))
    (fc1): Linear(in_features=1600, out_features=64, bias=True)
    (fc2): Linear(in_features=64, out_features=8, bias=True)
    (fc3): Linear(in_features=8, out_features=1, bias=True)
    (sig): Sigmoid()
    )
    ```
    ### Game visualization
    ```
    pong_utils.play(env, policy, time=100) 

    RESULTs:
    ------------
    x input - neural network
    tensor([[[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            ...,
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],

            [[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            ...,
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]]])

    x.shape  --> torch.Size([1, 2, 80, 80])

    ---

    x output - neural network
    tensor([[ 0.5336]])

    x.shape  torch.Size([1, 1])
    ```
    ![image9]

    ### Distributed computing - Collect trajectories in parallel (here: 4 agents)
    ```
    envs = pong_utils.parallelEnv('PongDeterministic-v4', n=4, seed=12345)
    prob, state, action, reward = pong_utils.collect_trajectories(envs, policy, tmax=5)

    print(reward)
    print()
    print('Length trajectory:', len(reward))

    RESULTS:
    ------------
    [array([ 0.,  0.,  0.,  0.]), array([ 0.,  0.,  0.,  0.]), array([ 0.,  0.,  0.,  0.]), array([ 0.,  0.,  0.,  0.]), array([ 0.,  0.,  0.,  0.])]

    Length trajectory: 5

    ```
    ### Surrogate Function for Training
    ```
    RIGHT=4
    LEFT=5
    # convert states to probability, passing through the policy
    def states_to_prob(policy, states):
        """ Convert states to probability
        
            INPUTS:
            -------------
                policy - (instance of Policy class) definition of a neural network
                states - (list of torch tensor) states[0] shape torch.Size([4, 2, 80, 80])
                        len of list = tmax from collecting trajectories          
            
            OUTPUTS:
            -------------
                policy_output - (torch tensor) shape torch.Size([tmax, 4]), tmax from collecting trajectories   
        """
        
        states = torch.stack(states)
        policy_input = states.view(-1,*states.shape[-3:])
        
        policy_output = policy(policy_input).view(states.shape[:-3])
        
        return policy_output

    def surrogate(policy, old_probs, states, actions, rewards,
                discount = 0.995, beta=0.01):
        """ 
            INPUTS:
            ------------
                policy - (instance of Policy class) definition of a neural network
                old_probs - (list of numpy arrays) 
                            like [array([ 0.47297633,  0.52704322,  0.52703786,  0.52704197], dtype=float32), array([...]), ...] 
                            len of list = tmax from collecting trajectories
                states - (list of torch tensor) states[0] shape torch.Size([4, 2, 80, 80])
                            len of list = tmax from collecting trajectories          
                actions - (list of numpy arrays)like [array([4, 5, 5, 5]), array([4, 4, 5, 5]), array([5, 4, 4, 5]), array([5, 4, 4, 4]), array([5, 4, 5, 5])]
                            actions[0] shape (4,)
                            len of list = tmax from collecting trajectories   
                rewards - (list of numpy arrays) like array([ 0.,  0.,  0.,  0.]), array([ 0.,  0.,  0.,  0.]), array([ 0.,  0.,  0.,  0.]), array([ 0.,  0.,  0.,  0.]), array([ 0.,  0.,  0.,  0.])]
                            rewards[0] shape (4,)
                            len of list = tmax from collecting trajectories   
                discount - (float) default = 0.995
                beta - (float) default = 0.01
            
            OUTPUTS:
            ------------
                surrogate - (torch tensor) like tensor(1.00000e-03 * 6.9168)
                            surrogate shape torch.Size([])
        """
        print('INPUT')
        print('-----------')
        print()
        print('policy', policy)
        print()
        print('old_probs', old_probs)
        print('old_probs len', len(old_probs))
        print()
        print('states', states)
        print('states len', len(states))
        print('states[0] shape', states[0].shape)
        print()
        print('actions', actions)
        print('actions len', len(actions))
        print('actions[0] shape', actions[0].shape)
        print()
        print('rewards', rewards)
        print('rewards len', len(rewards))
        print('rewards[0] shape', rewards[0].shape)
        print()
        print('discount', discount)
        print()
        print('beta', beta)
        print()

        # discount numpy array like [ 1.  0.995  0.990025  0.98507488  0.9801495 ] --> shape (tmax,)
        discount = discount**np.arange(len(rewards))
        print('discount', discount)
        print('discount shape', discount.shape)
        print()
        
        # rewards numpy array like
        # [[ 0.  0.  0.  0.]
        # [ 0.  0.  0.  0.]
        # [ 0.  0.  0.  0.]
        # [ 0.  0.  0.  0.]
        # [ 0.  0.  0.  0.]]
        # rewards shape (5, 4)
        rewards = np.asarray(rewards)*discount[:,np.newaxis]
        print('rewards', rewards)
        print('rewards shape', rewards.shape)
        print()
        
        # convert rewards to rewards_future like
        # [[ 0.  0.  0.  0.]
        # [ 0.  0.  0.  0.]
        # [ 0.  0.  0.  0.]
        # [ 0.  0.  0.  0.]
        # [ 0.  0.  0.  0.]]
        # rewards_future shape (5, 4)
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
        print('rewards_future', rewards_future)
        print('rewards_future shape', rewards_future.shape)
        print()
        
        # mean like [ 0.  0.  0.  0.  0.]
        # mean shape (5,)
        mean = np.mean(rewards_future, axis=1)
        print('mean', mean)
        print('mean shape', mean.shape)
        print()
        
        # std like [1.00000000e-10  1.00000000e-10  1.00000000e-10  1.00000000e-10  1.00000000e-10]
        # std shape (5,)
        std = np.std(rewards_future, axis=1) + 1.0e-10
        print('std', std)
        print('std shape', std.shape)
        print()

        # rewards_normalized [[ 0.  0.  0.  0.]
        # [ 0.  0.  0.  0.]
        # [ 0.  0.  0.  0.]
        # [ 0.  0.  0.  0.]
        # [ 0.  0.  0.  0.]]
        # rewards_normalized shape (5, 4)
        rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
        print('rewards_normalized', rewards_normalized)
        print('rewards_normalized shape', rewards_normalized.shape)
        print()
        
        # convert everything into pytorch tensors and move to gpu if available
        # actions like
        # tensor([[ 4,  5,  5,  5],
        #        [ 4,  4,  5,  5],
        #        [ 5,  4,  4,  5],
        #        [ 5,  4,  4,  4],
        #        [ 5,  4,  5,  5]], dtype=torch.int8)
        # actions shape torch.Size([5, 4])
        actions = torch.tensor(actions, dtype=torch.int8, device=device)
        
        # old_probs like
        # tensor([[ 0.4730,  0.5270,  0.5270,  0.5270],
        #        [ 0.4729,  0.4729,  0.5270,  0.5271],
        #        [ 0.5271,  0.4729,  0.4729,  0.5270],
        #        [ 0.5271,  0.4730,  0.4730,  0.4730],
        #        [ 0.5270,  0.4730,  0.5270,  0.5270]])
        # old_probs shape torch.Size([5, 4])
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        
        # rewards like
        # tensor([[ 0.,  0.,  0.,  0.],
        #        [ 0.,  0.,  0.,  0.],
        #        [ 0.,  0.,  0.,  0.],
        #        [ 0.,  0.,  0.,  0.],
        #        [ 0.,  0.,  0.,  0.]])
        # rewards shape torch.Size([5, 4])
        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)
        
        print('actions', actions)
        print('actions shape', actions.shape)
        print()
        print('old_probs', old_probs)
        print('old_probs shape', old_probs.shape)
        print()
        print('rewards', rewards)
        print('rewards shape', rewards.shape)
        print()
        
        # convert states to policy (or probability) like
        # new_probs  
        # tensor([[ 0.4730,  0.5270,  0.5270,  0.5270],
        #        [ 0.4729,  0.4729,  0.5270,  0.5271],
        #        [ 0.5271,  0.4729,  0.4729,  0.5270],
        #        [ 0.5271,  0.4730,  0.4730,  0.4730],
        #        [ 0.5270,  0.4730,  0.5270,  0.5270]])
        # new_probs shape torch.Size([5, 4])
        new_probs = states_to_prob(policy, states)
        new_probs = torch.where(actions == RIGHT, new_probs, 1.0-new_probs)
        
        print('new_probs ', new_probs)
        print('new_probs shape', new_probs.shape)
        print()
        
        # ratio like 
        # tensor([[ 1.0000,  1.0000,  1.0000,  1.0000],
        #        [ 1.0000,  1.0000,  1.0000,  1.0000],
        #        [ 1.0000,  1.0000,  1.0000,  1.0000],
        #        [ 1.0000,  1.0000,  1.0000,  1.0000],
        #        [ 1.0000,  1.0000,  1.0000,  1.0000]])
        # ratio shape torch.Size([5, 4])
        ratio = new_probs/old_probs
        
        print('ratio ', ratio)
        print('ratio shape', ratio.shape)
        print()

        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        
        # entropy like
        # tensor([[ 0.6917,  0.6917,  0.6917,  0.6917],
        #        [ 0.6917,  0.6917,  0.6917,  0.6917],
        #        [ 0.6917,  0.6917,  0.6917,  0.6917],
        #        [ 0.6917,  0.6917,  0.6917,  0.6917],
        #        [ 0.6917,  0.6917,  0.6917,  0.6917]])
        # entropy shape torch.Size([5, 4])
        entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
            (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))
        
        print('entropy ', entropy)
        print('entropy shape', entropy.shape)
        print()
        
        # surrogate  
        # tensor(1.00000e-03 * 6.9168)
        # surrogate shape torch.Size([])
        surrogate = torch.mean(ratio*rewards + beta*entropy)
        
        print('surrogate ', surrogate)
        print('surrogate shape', surrogate.shape)
        print()

        return surrogate


    Lsur= surrogate(policy, prob, state, action, reward)

    print(Lsur)
    ```
    ### Training 
    ```
    from parallelEnv import parallelEnv
    import numpy as np
    # WARNING: running through all 800 episodes will take 30-45 minutes

    # training loop max iterations
    episode = 100
    # episode = 800

    # widget bar to display progress
    !pip install progressbar
    import progressbar as pb
    widget = ['training loop: ', pb.Percentage(), ' ', 
            pb.Bar(), ' ', pb.ETA() ]
    timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

    # initialize environment
    envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

    discount_rate = .99
    beta = .01
    tmax = 320

    # keep track of progress
    mean_rewards = []

    for e in range(episode):

        # collect trajectories
        old_probs, states, actions, rewards = \
            pong_utils.collect_trajectories(envs, policy, tmax=tmax)
            
        total_rewards = np.sum(rewards, axis=0)

        # this is the SOLUTION!
        # use your own surrogate function
        #L = -pong_utils.surrogate(policy, old_probs, states, actions, rewards, beta=beta)
        
        L = -surrogate(policy, old_probs, states, actions, rewards, beta=beta)
        
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        del L
            
        # the regulation term also reduces
        # this reduces exploration in later runs
        beta*=.995
        
        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))
        
        # display some progress every 20 iterations
        if (e+1)%20 ==0 :
            print("Episode: {0:d}, score: {1:f}".format(e+1,np.mean(total_rewards)))
            print(total_rewards)
            
        # update progress widget bar
        timer.update(e+1)
        
    timer.finish()
    ```

## Importance Sampling <a name="imp_samp"></a> 
- How does a normal policy Update work in REINFORCE so far?

    ![image11]

- How to reuse important trajectories?

    ![image10]

## Proximal Policy Optimization - PPO <a name="ppo"></a> 
- How does Proximal Policy Optimization work?

    ![image12]

- The Surrogate Function

    ![image13]

## PPO with Clipping - PPO <a name="ppo_clip"></a> 
- **Clip** the Surrogate function to ensure that **the new policy remains close to the old one**
- **Continually updating the policy** via gradient ascent could lead to a **cliff** --> The Policy/Reward Cliff
- This could lead to a really bad policy that is very hard to recover from

    ![image14]

- How to fix this?

    ![image15]

### Summary
- We can finally summarize the PPO algorithm

    ![image16]

## Pong with PPO <a name="pong_ppo"></a> 
- Open Jupyter Notebook ```pong_ppo.ipynb```
    ### Install package for displaying animation
    ```
    # install package for displaying animation
    !pip install JSAnimation

    # custom utilies for displaying animation, collecting rollouts and more
    import pong_utils

    %matplotlib inline

    # check which device is being used. 
    # I recommend disabling gpu until you've made sure that the code runs
    device = pong_utils.device
    print("using device: ",device)
    ```
    ### Render AI gym environment 
    ```
    # render ai gym environment
    import gym
    import time

    # PongDeterministic does not contain random frameskip
    # so is faster to train than the vanilla Pong-v4 environment
    env = gym.make('PongDeterministic-v4')

    print("List of available actions: ", env.unwrapped.get_action_meanings())

    # we will only use the actions 'RIGHTFIRE' = 4 and 'LEFTFIRE" = 5
    # the 'FIRE' part ensures that the game starts again after losing a life
    # the actions are hard-coded in pong_utils.py

    RESULTS:
    ------------
    List of available actions:  ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
    ```
    ### Preprocessing
    ```
    import matplotlib
    import matplotlib.pyplot as plt

    # show what a preprocessed image looks like
    env.reset()
    _, _, _, _ = env.step(0)
    # get a frame after 20 steps
    for _ in range(20):
        frame, _, _, _ = env.step(1)

    plt.subplot(1,2,1)
    plt.imshow(frame)
    plt.title('original image')

    plt.subplot(1,2,2)
    plt.title('preprocessed image')

    # 80 x 80 black and white image
    plt.imshow(pong_utils.preprocess_single(frame), cmap='Greys')
    plt.show()
    ```
    ![image8]
    ### Implement a policy
    ```
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # set up a convolutional neural net
    # the output is the probability of moving right
    # P(left) = 1-P(right)
    class Policy(nn.Module):
        """ Define a policy based on a neural network
        """

        def __init__(self):
            """ Init of class Policy()
            
                INPUTS:
                ------------
                    None
                    
                OUTPUTS:
                ------------
                    No direct
            """
            
            super(Policy, self).__init__()
            
            # 80x80 to outputsize x outputsize
            # outputsize = (inputsize - kernel_size + stride)/stride 
            # (round up if not an integer)

            # conv1 80x80 --> 40x40 (due to stride=2)
            self.conv1 = nn.Conv2d(2, 4, kernel_size=2, stride=2)
            # conv2 40x40 --> 20x20 (due to stride=2)
            self.conv2 = nn.Conv2d(4, 8, kernel_size=2, stride=2)
            # conv3 20x20 --> 10x10 (due to stride=2)
            self.conv3 = nn.Conv2d(8, 16, kernel_size=2, stride=2)
            self.size=16*10*10
            
            # 3 fully connected layer
            self.fc1 = nn.Linear(self.size, 64)
            self.fc2 = nn.Linear(64, 8)
            self.fc3 = nn.Linear(8, 1)
            self.sig = nn.Sigmoid()
            
        def forward(self, x):
            """ Forward path of neural network
            
                INPUTS:
                ------------
                    x - (torch tensor) shape torch.Size([1, 2, 80, 80]) 
                        1 --> number of parallel instances, up to 4 at the moment
                        2 --> two consecutive stacked frames
                        80x80 --> pixel width and height
                    
                OUTPUTS:
                ------------
                    x - (torch tensor) shape torch.Size([1, 1]) - Probability to choose RIGHT as action
            """
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            
            # flatten the tensor
            x = x.view(-1,self.size)
            
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.sig(self.fc3(x))
            return x

    # use your own policy!
    policy=Policy().to(device)
    print(policy)

    #policy=pong_utils.Policy().to(device)

    # we use the adam optimizer with learning rate 2e-4
    # optim.SGD is also possible
    import torch.optim as optim
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    RESULTS:
    ------------
    Policy(
    (conv1): Conv2d(2, 4, kernel_size=(2, 2), stride=(2, 2))
    (conv2): Conv2d(4, 8, kernel_size=(2, 2), stride=(2, 2))
    (conv3): Conv2d(8, 16, kernel_size=(2, 2), stride=(2, 2))
    (fc1): Linear(in_features=1600, out_features=64, bias=True)
    (fc2): Linear(in_features=64, out_features=8, bias=True)
    (fc3): Linear(in_features=8, out_features=1, bias=True)
    (sig): Sigmoid()
    )
    ```
    ### Game visualization
    ```
    pong_utils.play(env, policy, time=100) 

    RESULTs:
    ------------
    x input - neural network
    tensor([[[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            ...,
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],

            [[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            ...,
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]]])

    x.shape  --> torch.Size([1, 2, 80, 80])

    ---

    x output - neural network
    tensor([[ 0.5336]])

    x.shape  torch.Size([1, 1])
    ```
    ![image9]

    ### Distributed computing - Collect trajectories in parallel (here: 4 agents)
    ```
    envs = pong_utils.parallelEnv('PongDeterministic-v4', n=4, seed=12345)
    prob, state, action, reward = pong_utils.collect_trajectories(envs, policy, tmax=5)

    print(reward)
    print()
    print('Length trajectory:', len(reward))

    RESULTS:
    ------------
    [array([ 0.,  0.,  0.,  0.]), array([ 0.,  0.,  0.,  0.]), array([ 0.,  0.,  0.,  0.]), array([ 0.,  0.,  0.,  0.]), array([ 0.,  0.,  0.,  0.])]

    Length trajectory: 5

    ```
    ### PPO
    ```
    RIGHT=4
    LEFT=5
    # convert states to probability, passing through the policy
    def states_to_prob(policy, states):
        """ Convert states to probability

            INPUTS:
            -------------
                policy - (instance of Policy class) definition of a neural network
                states - (list of torch tensor) states[0] shape torch.Size([4, 2, 80, 80])
                        len of list = tmax from collecting trajectories          

            OUTPUTS:
            -------------
                policy_output - (torch tensor) shape torch.Size([tmax, 4]), tmax from collecting trajectories   
        """

        states = torch.stack(states)
        policy_input = states.view(-1,*states.shape[-3:])

        policy_output = policy(policy_input).view(states.shape[:-3])

        return policy_output


    # clipped surrogate function
    # similar as -policy_loss for REINFORCE, but for PPO
    def clipped_surrogate(policy, old_probs, states, actions, rewards, discount=0.995, epsilon=0.1, beta=0.01):
        """ Clipped Surrogate Function 
        
            INPUTS:
            ------------
                policy - (instance of Policy class) definition of a neural network
                old_probs - (list of numpy arrays) 
                            like [array([ 0.47297633,  0.52704322,  0.52703786,  0.52704197], dtype=float32), array([...]), ...] 
                            len of list = tmax from collecting trajectories
                states - (list of torch tensor) states[0] shape torch.Size([4, 2, 80, 80])
                            len of list = tmax from collecting trajectories          
                actions - (list of numpy arrays)like [array([4, 5, 5, 5]), array([4, 4, 5, 5]), array([5, 4, 4, 5]), array([5, 4, 4, 4]), array([5, 4, 5, 5])]
                            actions[0] shape (4,)
                            len of list = tmax from collecting trajectories   
                rewards - (list of numpy arrays) like array([ 0.,  0.,  0.,  0.]), array([ 0.,  0.,  0.,  0.]), array([ 0.,  0.,  0.,  0.]), array([ 0.,  0.,  0.,  0.]), array([ 0.,  0.,  0.,  0.])]
                            rewards[0] shape (4,)
                            len of list = tmax from collecting trajectories   
                discount - (float) default = 0.995
                epsilon - (float) to define a threshold (1 + epsilon) for activating clipping
                beta - (float) default = 0.01

            OUTPUTS:
            ------------
                surrogate - (torch tensor) like tensor(1.00000e-03 * 6.9168)
                            surrogate shape torch.Size([])
        """

        discount = discount**np.arange(len(rewards))
        rewards = np.asarray(rewards)*discount[:,np.newaxis]
        
        # convert rewards to future rewards
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
        
        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10

        rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
        
        # convert everything into pytorch tensors and move to gpu if available
        actions = torch.tensor(actions, dtype=torch.int8, device=device)
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

        # convert states to policy (or probability)
        new_probs = states_to_prob(policy, states)
        new_probs = torch.where(actions == RIGHT, new_probs, 1.0-new_probs)
        
        # ratio for clipping
        ratio = new_probs/old_probs

        # clipped function
        clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
            (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

        
        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        return torch.mean(clipped_surrogate + beta*entropy)
    ```
    ### Training
    ```
    from parallelEnv import parallelEnv
    import numpy as np
    # keep track of how long training takes
    # WARNING: running through all 800 episodes will take 30-45 minutes

    # training loop max iterations
    episode = 100

    # widget bar to display progress
    !pip install progressbar
    import progressbar as pb
    widget = ['training loop: ', pb.Percentage(), ' ', 
            pb.Bar(), ' ', pb.ETA() ]
    timer = pb.ProgressBar(widgets=widget, maxval=episode).start()


    envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

    discount_rate = .99
    epsilon = 0.1
    beta = .01
    tmax = 320
    SGD_epoch = 4

    # keep track of progress
    mean_rewards = []

    for e in range(episode):

        # collect trajectories
        old_probs, states, actions, rewards = \
            pong_utils.collect_trajectories(envs, policy, tmax=tmax)
            
        total_rewards = np.sum(rewards, axis=0)


        # gradient ascent step
        for _ in range(SGD_epoch):
            
            # L = -pong_utils.clipped_surrogate(policy, old_probs, states, actions, rewards, epsilon=epsilon, beta=beta)
            
            # uncomment to utilize your own clipped function!
            L = -clipped_surrogate(policy, old_probs, states, actions, rewards, epsilon=epsilon, beta=beta)

            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            del L
        
        # the clipping parameter reduces as time goes on
        epsilon*=.999
        
        # the regulation term also reduces
        # this reduces exploration in later runs
        beta*=.995
        
        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))
        
        # display some progress every 20 iterations
        if (e+1)%20 ==0 :
            print("Episode: {0:d}, score: {1:f}".format(e+1,np.mean(total_rewards)))
            print(total_rewards)
            
        # update progress widget bar
        timer.update(e+1)
        
    timer.finish()
    ```
    ### Watch a trained Agent 
    ```
    pong_utils.play(env, policy, time=200) 
    ```
    ### Save Policy
    ```
    # save your policy!
    torch.save(policy, 'PPO.policy')

    # load policy if needed
    # policy = torch.load('PPO.policy')

    # try and test out the solution 
    # make sure GPU is enabled, otherwise loading will fail
    # (the PPO verion can win more often than not)!
    #
    # policy_solution = torch.load('PPO_solution.policy')
    # pong_utils.play(env, policy_solution, time=2000) 
    ```


## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Sparkify-Project.git
```

- Change Directory
```
$ cd Sparkify-Project
```

- Create a new Python environment, e.g. spark_env. Inside Git Bash (Terminal) write:
```
$ conda create --name spark_env
```

- Activate the installed environment via
```
$ conda activate spark_env
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
pyspark = 2.4.3
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Further Deep Reinforcement Learning References
* [Very good summary of DQN](https://medium.com/@nisheed/udacity-deep-reinforcement-learning-project-1-navigation-d16b43793af5)
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)
