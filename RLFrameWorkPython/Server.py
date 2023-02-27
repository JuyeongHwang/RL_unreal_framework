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

import gym

class Actor(Model):
    def __init__(self, action_dim, action_bound):
        super(Actor,self).__init__()
        self.action_bound = action_bound

        self.h1 = Dense(64,activation='relu')
        self.h2 = Dense(32,activation='relu')
        self.h3 = Dense(16,activation='relu')
        self.mu = Dense(action_dim,activation='tanh') # output 평균
        self.std = Dense(action_dim, activation='softplus') # output 표준편차

    def call(self,state):

        x = self.h1(state) #input
        x = self.h2(x)
        x = self.h3(x)

        mu = self.mu(x)
        std = self.std(x)

        #평균값을 조절
        mu = Lambda(lambda x : x*self.action_bound)(mu)
        return [mu,std]
class PPOAgent(object):

    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound

        #액터 신경망 및 크리틱 신경망 생성.
        self.actor = Actor(self.action_dim, self.action_bound)
        self.actor.build(input_shape=(None,self.state_dim))
        #에피소드로부터 얻은 총 보상값 저장
        self.save_epi_reward = []
    def get_policy_action(self,state):
        mu_a,std_a = self.actor(state)

        mu_a = mu_a.numpy()[0]
        std_a = std_a.numpy()[0]
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size = self.action_dim)

        return mu_a, std_a, action

# ===== train region =====
batch_state, batch_action,batch_reward = [],[],[]
batch_log_old_policy_pdf = []

agent = PPOAgent(4, 1, 50.0, [1e-2, 1.0])


# ========================
print("... Complete Make PPO Model ...")

HEADER = 1024
PORT = 5053
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
    states = []
    while connected:
        msg_length = connection.recv(HEADER).decode(FORMAT)

        if msg_length: #not none

            # ======= state block ===========
            texts = msg_length.split("/")
            # print(len(texts)," ", texts)
            for i in range(len(texts)):
                if (texts[i] == ""): pass
                if(texts[i] == "state"):
                    startState = True

                if (startState and is_number(texts[i])):
                    states.append(float(texts[i]))
                if(texts[i]=="stateend"):
                    endState = True
                    startState = False

            # print(f"[{address}] {msg_length}")
            if(endState==True and startState == False):
                for i in range(len(states)):
                    print(f"{states[i]}")
                endState = False
                mu_old, std_old, action = agent.get_policy_action(tf.convert_to_tensor([states], dtype=tf.float32))
                # connection.send(bytes("action/ ",'utf-8'))
                sendMsgFloat(connection, action[0])
                # sendMsgFloat(connection, action[1])
                print(f"action {action}")
                states.clear()

            # ======= state ===========

    connection.close()


def train(mu_old, std_old, action):

    # 이전 정책의 로그 확률밀도함수 계산
    var_old = std_old ** 2
    log_old_policy_pdf = -0.5 * (action - mu_old) ** 2 / var_old - 0.5 * np.log(var_old * 2 * np.pi)
    log_old_policy_pdf = np.sum(log_old_policy_pdf)


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
