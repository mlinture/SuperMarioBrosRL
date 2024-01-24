FIRST:  
From the home directory (SuperMarioBrosRL) run:  
$ python3 -m venv .venv  
$ source .venv/bin/activate  
$ pip3 install -r requirements.txt  

Then, cd into src.  
In our project there are three different models:  

1. DQN  
  - For the regular DQN model, there are two main scripts: train.py and play.py  

  train.py:
  - To run train.py, run the command 'python3 -m supermario_dqn.cmds.train'
    - In line 66 of the script, there is an option for you to set a destination path, where upon completion of training, the model used will be saved to that pathname. Create a file of your choice and set the save_path variable to its string.
    - If you would like to adjust any hyperparameters, it can be done by manually going through the main function's argparse default values and hard coding whatever value you see fit for each variable.

  play.py:
  - To run play.py, run the command 'python3 -m supermario_dqn.cmds.play'
  - In the main function of the file (line 13), there is a parameter 'model' which is default set to None. If you have a previously trained model you would like to use to play the game, you can hardcode to set the default parameter to be the string of the pathname of the saved model.

2. Dueling DQN
  - For the dueling DQN model, there is one executable duel_train.py

  duel_train.py:
  - To run duel_train.py, run the command 'python3 -m supermario_dqn.cmds.duel_train'
    - In line 66 of the script, there is an option for you to set a destination path, where upon completion of training, the model used will be saved to that pathname. Create a file of your choice and set the save_path variable to its string.
    - If you would like to adjust any hyperparameters, it can be done by manually going through the main function's argparse default values and hard coding whatever value you see fit for each variable.

3. PPO
  - For the PPO algorithm, there is one executable train_ppo.py

  train_ppo.py:   
  BEFORE RUNNING  
    - In the same directory as train_ppo.py (which should be ppo_fork), create two folders titled 'logs' and 'train'. These files will be updated periodically during training to store statistics regarding the model and the model itself, respectively.
  - If you would like to train, run the script as is; cd into ppo_fork, then run the command 'python3 train_ppo'.
  - If you would like to play, open train_ppo.py. Replace line 57 with 'play(YOUR_SAVED_ZIP_FILE, environment).
    - For example, if you had a file saved in logs titled model_40000.zip, the command you would run is 'play(model_40000, environment)'.

As dicussed in our final presentation, there may be issues when running the above mentioned models. This is due to the discrepancies between the venv files from our initial fork and when we downloaded them locally in our first step.

The best advice we can give is to follow each error message; if the message follows the lines of "seed/options doesn't exist", continue navigating through each file and delete the seed or options parameters until the scripts start working.

In ./SuperMarioBrosRL/.venv/lib/python3.11/site-packages/gym/wrappers/time_limit.py, change the step function starting at line 39 to:

      def step(self, action):
        """Steps through the environment and if the number of steps elapsed exceeds ``max_episode_steps`` then truncate.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, terminated, truncated, info)`` with `truncated=True`
            if the number of steps elapsed >= max episode steps

        """
        hold = self.env.step(action)
        if len(hold)==5:
            observation, reward, terminated, truncated, info = hold
            self._elapsed_steps += 1

            #if self._elapsed_steps >= self._max_episode_steps:
            #    truncated = True

            return observation, reward, terminated, truncated, info
        else:
            observation, reward, terminated, info = hold
            self._elapsed_steps += 1

            #if self._elapsed_steps >= self._max_episode_steps:
            #    truncated = True

            return observation, reward, terminated, info

        
If there are any inquiries or issues running our repository, please do not hesitate to contact us at the emails written at the top of this document!
