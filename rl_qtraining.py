import re
import numpy as np
import os
import sys
#if using Theano with GPU
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import random
from keras.models import Sequential
from keras.layers.core import Dense
from collections import deque

class Game():
    def reset(self, new_example, true_acc, is_clapping): # new_example = frames*(v,d), v = vertices of 13 parts, d = vel of 13 parts
        self.state = new_example # store v and d in one tensor (60frames*13joints*2features*3axis)
        self.frame = 0
        self.is_clapping = is_clapping
        self.true_acc = true_acc # store true acc in one tensor (60frames*13joints*1feature*3axis)
        self.max_frame = self.state.shape[0]-1
        
        if self.is_clapping == True:
            self.scaling_factor = 1
        else:
            self.scaling_factor = 0.25
        
    def flatten(self):
        # flatten the vd matrix size=(13*2*3) into 1 dimension size=(1*78)
        # flatten the a matrix size=(13*1*3) into 1 dimension size=(1*39)
        flatten_state = self.state[self.frame].reshape((1,-1)).copy()
        flatten_acc = self.true_acc[self.frame].reshape((1,-1)).copy()
        self.flatten_state = flatten_state
        self.flatten_acc = flatten_acc
#         print(flatten_state.shape)
        
        return flatten_state, flatten_acc
    
    def update(self):
        self.frame += 1 # move to next frame
        if self.is_clapping:
            return 1
        else:
            return -1


class Agent():
    def __init__(self, env, explore=0.1, discount=0.95, hidden_size=10, memory_limit=5000):
        self.env = env
        model = Sequential()
        model.add(Dense(hidden_size, input_shape=(2*13*3,), activation='relu'))
        model.add(Dense(hidden_size, activation='relu'))
        model.add(Dense(13*3))
        model.compile(loss='mse', optimizer='sgd')
        # model.compile(loss='mse', optimizer='sgd', sample_weight_mode='temporal')
        self.Q = model
        
        # experience replay:
        # remember states to "reflect" on later
        self.memory = deque([], maxlen=memory_limit)
        
        self.explore = explore # give some chance to randomly assign acc to explore new possiblities
        self.discount = discount # gamma of weight
    
    def predict_acc(self):
        if np.random.rand() <= self.explore:
            return np.random.rand(1, 13)
        state = self.env.flatten_state
        q = self.Q.predict(state)
        
        return q[0]
    
    def remember(self, state, true_acc, frame, reward):
        # the deque object will automatically keep a fixed length
        self.memory.append((state, true_acc, frame, reward))
    
    
    # between every action, it will pick batch_size previous states/actions that the agent "remembers" to train
    def _prep_batch(self, batch_size):
        if batch_size > self.memory.maxlen:
            Warning('batch size should not be larger than max memory size. Setting batch size to memory size')
            batch_size = self.memory.maxlen
            
        batch_size = min(batch_size, len(self.memory))
        
        inputs = []
        targets = []
        rewards = []
        
        # prep the batch
        # inputs are states, outputs are values over actions
        batch = random.sample(list(self.memory), batch_size) # randomly get some states in the memory
        random.shuffle(batch)
        
        for state, true_acc, frame, reward in batch:
            inputs.append(state)
            targets.append(true_acc)
            rewards.append([[frame], [self.env.scaling_factor * pow(self.discount,self.env.max_frame - frame)*reward]])
        # to numpy matrices
        return np.vstack(inputs), np.vstack(targets), np.vstack(rewards)
    
    def replay(self, batch_size):
        inputs, targets, rewards= self._prep_batch(batch_size)
        # the model is trained here
        print(rewards.shape)
        # loss = self.Q.train_on_batch(inputs, targets, sample_weight=rewards)
        loss = self.Q.train_on_batch(inputs, targets)
        return loss
    
    def save(self, fname):
        self.Q.save_weights(fname)
    
    def load(self, fname):
        self.Q.load_weights(fname)
#         print(self.Q.get_weights())




def train(filefolder):
    game = Game()
    agent = Agent(game)

    # get .npz filenames from a folder
    files = [filefolder+file for file in os.listdir(filefolder) if re.search(".npz", file)]


    print('training...')
    # epochs = 100 # number of training videos
    batch_size = 1000 # number of "experiences" we want to train from the previous decisions
    fname = 'clapping_weights.h5'

    # keep track of past record_len results
    record_len = 100
    record = deque([], record_len)

    for file in files:
        with np.load(file, 'r') as data:
            new_example = data['st']
            true_acc = data['ac']

        if re.search("non", file):
            is_clapping = False
        else:
            is_clapping = True

        print("{}: vd shape: {} acc shape: {}".format(file, new_example.shape, true_acc.shape))

        game.reset(new_example, true_acc, is_clapping)
        reward = 0
        loss = 0

        # rewards only given at end of game
        while game.frame != game.max_frame:
            prev_state, orig_acc = game.flatten()
            acc = agent.predict_acc()
            reward = game.update()

    #         acc.reshape((1,39))
    #         print("aaa")
    #         print(orig_acc.shape)
    #         print(acc.shape)
            agent.remember(prev_state, orig_acc, game.frame-1, reward)

            # the process will be trained in replay()
        loss = agent.replay(batch_size)

        print("loss: {}".format(loss))

    agent.save(fname) 


def evaluate():
    game = Game()
    agent = Agent(game)
    agent.load("clapping_weights.h5")
    # get .npz filenames from a folder
    filefolder = 'output/'
    files = [filefolder+file for file in os.listdir(filefolder) if re.search(".npz", file)]
    with np.load(files[0], 'r') as data:
        new_example = data['st']
        true_acc = data['ac']

    if re.search("non", files[0]):
        is_clapping = False
    else:
        is_clapping = True
    game.reset(new_example, true_acc, is_clapping)
    
    predicted_v = []
    prev_state, orig_acc = game.flatten()
    while game.frame != game.max_frame:
        acc = agent.predict_acc()
        if len(acc) != 39:
            acc = np.hstack((acc, np.zeros((1,26))))
        v, d = np.hsplit(game.flatten_state, 2)
        predicted_v.append(v.reshape((13,3)).copy())
        v.reshape((1,-1))
        d.reshape((1,-1))
        d = d + acc
        v = v + d
        reward = game.update()
        game.flatten_state = np.hstack((v, d))
    
    
    result = np.asarray(predicted_v)
    np.save("result.npy", result)

if __name__ == "__main__":
    # filefolder = str(sys.argv[1])
    # train(filefolder)
    evaluate()