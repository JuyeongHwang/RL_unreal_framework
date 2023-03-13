import  socket
import threading
import gym
import numpy as np

def is_number(value) :
    try :
        float(value)
        return True
    except ValueError :
        return False

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Lambda
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 3


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
# ===== train region =====
action_bound = 10
state_dim = 4
action_dim = 1
model = PPO()
# model.load_state_dict(torch.load("./save_weights/cartpole_ppo4.h5"))
# ========================
print("... Complete Make PPO Model ...")

HEADER = 1024
PORT = 5050
SERVER = "127.0.0.1" # socket.gethostbyname(socket.gethostname())
ADDRESS = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECTED_MSG = "!DISCONNECT!"


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDRESS)

def handle_client(connection, address):
    print(f"[New CONNECTION] {address} connected.")
    connected = True
    startState = False
    endState = False

    startObs = False
    endObs = False

    state = []
    obsCount = 0
    next_state = []
    reward =0
    done = 0
    action = 0
    episode = 0
    while connected:
        if(episode>2000):
            print("...Episode Exit...")
            torch.save(model.state_dict(), "./save_weights/cartpole_ppo4.h5")
            break

        msg_length = connection.recv(HEADER).decode(FORMAT)
        if msg_length: #not none

            #handle text
            texts = msg_length.split("/")
            # print(len(texts)," ", texts)
            for i in range(len(texts)):
                # if (texts[i] == ""): pass
                # ======= state region ===========
                if(texts[i] == "state"):
                    startState = True
                    state = []
                if(startState and is_number(texts[i])):
                    state.append(float(texts[i])/10.0)
                if(texts[i]=="stateend"):
                    endState = True
                    startState = False
                # ======= region end ===========
                # ======= observation region ===========
                if(texts[i] == "obs"):
                    startObs = True
                    next_state = []
                    reward = 0
                    done = 0
                if(startObs and is_number(texts[i])):
                    if(obsCount <= 3):
                        next_state.append(float(texts[i])/10.0)
                    elif(obsCount==4):
                        reward = float(texts[i])
                    else:
                        done = float(texts[i])
                    obsCount +=1
                if(texts[i]=="obsend"):
                    endObs = True
                    startObs = False
                    obsCount = 0
                # ======= region end ===========

        # print(f"[{address}] {msg_length}")
        if (endState == True and startState == False):
            # for i in range(len(state)):
            # print(f"{state}")
            endState = False
            prob = model.pi(torch.from_numpy(np.array(state)).float())
            m = Categorical(prob)
            action = m.sample().item()
            # connection.send(bytes("action/ ",'utf-8'))
            sendMsgFloat(connection, action)


        if (endObs == True and startObs == False):
            # print("==Observation complete==")
            endObs = False
            if(len(state)!=0):
                # print("==Put complete==")
                model.put_data((state, action, reward, next_state, prob[action].item(), done))
            # print(f"{state},{action},{reward},{next_state},{prob[action].item()},{done}")
            # print(f"action {action}")
            #print(f"state {state}")
            #print(f"reward {reward}")
            #print(f"next_state {next_state}")
            #print(f"next_state {next_state}")
            #print(f"prob[action].item() {prob[action].item()}")
            #print(f"done {done}")

            if(done==1.0):
                if(episode%32==0):
                    model.train_net()
                    print("==Trained complete==")
                if(episode%64==0):
                    torch.save(model.state_dict(), "./save_weights/cartpole_ppo4.h5")
                print(f"episode : {episode}")
                episode+=1

    connection.close()



def sendMsgFloat(connection,fmsg):
    send = bytes(str(fmsg), 'utf-8')
    connection.send(send)

def start():
    server.listen() # listening new connection
    print(f"[LISTENING] Server is listening on {SERVER}\n")
    while True:
        connection,address = server.accept()
        thread = threading.Thread(target=handle_client, args=(connection,address))
        thread.start()
        connection.send(b"I'm Server.")
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() -1}\n")


print("[STARTING] server is starting ...")
start()


'''
                # GAE와 시간차 타깃 계산
                next_v_value = agent.critic(tf.convert_to_tensor([next_state], dtype=tf.float32))
                v_values = agent.critic(tf.convert_to_tensor(states, dtype=tf.float32))
                gaes, y_i = agent.gae_target(rewards, v_values.numpy(), next_v_value.numpy(), done)

                # epoch만큼 반복
                for _ in range(10):
                    # 액터 신경망 업데이트
                    agent.actor_learn(tf.convert_to_tensor(log_old_policy_pdfs, dtype=tf.float32),
                                     tf.convert_to_tensor(states, dtype=tf.float32),
                                     tf.convert_to_tensor(actions, dtype=tf.float32),
                                     tf.convert_to_tensor(gaes, dtype=tf.float32))

                    agent.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))
'''
