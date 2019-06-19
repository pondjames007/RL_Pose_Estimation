# Reinforcement Learning Multi-Person Pose Estimation
Final Project for Vision Meets Machine Learning
Trying to apply Q-learning on Multi-Person Pose Estimation
The goal is to use Q-learning to recognize the human action **Clapping**
It is based on the paper [CVPR'17 paper](https://arxiv.org/abs/1611.08050) and its [GitHub link](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/README.md).

The human action detection dataset is from [NTU RGB+D dataset](https://github.com/shahroudy/NTURGB-D).

## Codes
#### read_skeleton_file.py
Arrange skeleton data to what we need, discard some joints we don't need.

#### rl_qtraining.py
Apply Q-Learning on to the data, train the agent to recognize clapping.

## Approaches
* Use Traditional Q-Learning
* Setup a game (move all skeletons) and an agent (try to do clapping)
* Inputs: States (13 joints * 2 features(v, d) * 3 coordinates)
* Outputs: Actions (13 * 1 * 3)
* Model
``` 
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(2*13*3,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(13*3))
    model.compile(loss='mse', optimizer='sgd')
    self.Q = model 
```

* Agents remembers previous moves
* Give reward for each action according to the frame number and do training at the end

``` rewards.append([[frame], [self.env.scaling_factor * pow(self.discount,self.env.max_frame - frame)*reward]]) ```

* Gamma (discount) = 0.95
* Training
```
    def replay(self, batch_size):
        inputs, targets, rewards= self._prep_batch(batch_size)
        # the model is trained here
        print(rewards.shape)
        loss = self.Q.train_on_batch(inputs, targets)
        return loss
```
