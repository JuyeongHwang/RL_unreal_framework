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
class Critic(Model):
    def __init__(self):
        super(Critic,self).__init__()

        self.h1 = Dense(64,activation='relu')
        self.h2 = Dense(32,activation='relu')
        self.h3 = Dense(16,activation='relu')
        self.v = Dense(1,activation='linear')

    def call(self,state):

        x = self.h1(state) #input
        x = self.h2(x)
        x = self.h3(x)

        v = self.v(x)

        return v
class PPOAgent(object):

    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.GAMMA = 0.90
        self.GAE_LAMBDA = 0.9
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.RATIO_CLIPPING = 0.2

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound

        #액터 신경망 및 크리틱 신경망 생성.
        self.actor = Actor(self.action_dim, self.action_bound)
        self.actor.build(input_shape=(None,self.state_dim))
        self.critic = Critic()
        self.critic.build(input_shape=(None,self.state_dim))

        # 옵티마이저
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(self.CRITIC_LEARNING_RATE)
    def get_policy_action(self,state):
        mu_a,std_a = self.actor(state)

        mu_a = mu_a.numpy()[0]
        std_a = std_a.numpy()[0]
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size = self.action_dim)

        return mu_a, std_a, action
    def unpack_batch(self,batch):
        unpack = batch[0]
        for idx in range(len(batch) - 1):
            unpack = np.append(unpack, batch[idx + 1], axis=0)

        return unpack
    def gae_target(self,rewards, v_values, next_v_value,done):
        n_step_targets = np.zeros_like(rewards) #rewards length만큼 0.0으로 채워진 배열 만들어짐
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0,len(rewards))):
            delta = rewards[k] + self.GAMMA * forward_val - v_values[k]
            gae_cumulative = self.GAMMA*self.GAE_LAMBDA *gae_cumulative+delta
            gae[k] =  gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]

        return gae, n_step_targets

    def log_pdf(self,mu,std,action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std**2
        log_policy_pdf = -0.5*(action-mu)**2/var-0.5*tf.math.log(var*2*np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def actor_learn(self,log_old_policy_pdf, states,actions,gaes):

        with tf.GradientTape() as tape:
            mu_a, std_a = self.actor(states, training = True)
            log_policy_pdf= self.log_pdf(mu_a, std_a, actions)
            ratio = tf.exp(log_policy_pdf - log_old_policy_pdf)
            clipped_ratio = tf.clip_by_value(ratio, 1.0-self.RATIO_CLIPPING, 1.0+self.RATIO_CLIPPING)
            surrogate = -tf.minimum(ratio*gaes, clipped_ratio*gaes)
            loss = tf.reduce_mean(surrogate) #전체 평균

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

    def critic_learn(self,states,td_targets):

        with tf.GradientTape() as tape:
            td_hat = self.critic(states,training = True)
            loss = tf.reduce_mean(tf.square(td_hat - td_targets))
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

# ===== train region =====
action_bound = 10
state_dim = 4
action_dim = 1
agent = PPOAgent(state_dim, action_dim, action_bound, [1e-2, 1.0])


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

    batch_state, batch_action, batch_reward = [], [], []
    batch_log_old_policy_pdf = []

    while connected:
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
                    state.append(float(texts[i]))
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
                        next_state.append(float(texts[i]))
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
            #    print(f"{state[i]}")
            endState = False
            mu_old, std_old, action = agent.get_policy_action(tf.convert_to_tensor([state], dtype=tf.float32))
            action = np.clip(action, -action_bound, action_bound)

            action = action*10
            # 이전 정책의 로그 확률밀도함수 계산
            var_old = std_old ** 2
            log_old_policy_pdf = -0.5 * (action - mu_old) ** 2 / var_old - 0.5 * np.log(var_old * 2 * np.pi)
            log_old_policy_pdf = np.sum(log_old_policy_pdf)

            # connection.send(bytes("action/ ",'utf-8'))
            sendMsgFloat(connection, action[0])
            # sendMsgFloat(connection, action[1])
            print(f"action {action}")
            state = np.reshape(state, [1, state_dim])
            action = np.reshape(action, [1, action_dim])
            log_old_policy_pdf = np.reshape(log_old_policy_pdf, [1, 1])
            batch_state.append(state)
            batch_action.append(action)
            batch_log_old_policy_pdf.append(log_old_policy_pdf)

        if (endObs == True and startObs == False):
            print("==Observation complete==")
            endObs = False
            print(f"Reward {reward}")
            reward = np.reshape(reward, [1, 1])
            batch_reward.append(reward)

        if (len(batch_state) > 255 and len(batch_reward) > 255):
            print(f"==update network!==")
            connection.send(bytes("/Delay", 'utf-8'))
            # 배치가 채워지면 학습진행
            # 배치에서 데이터 추출
            states = agent.unpack_batch(batch_state)
            actions = agent.unpack_batch(batch_action)
            rewards = agent.unpack_batch(batch_reward)
            log_old_policy_pdfs = agent.unpack_batch(batch_log_old_policy_pdf)
            # print(f"state : {len(states)},actions : {len(actions)},rewards : {len(rewards)},log_old_policy_pdfs : {len(log_old_policy_pdfs)}")
            # 배치 비움
            batch_state, batch_action, batch_reward = [], [], []
            batch_log_old_policy_pdf = []

            # print(f"Next state : {len(next_state)}")
            # GAE와 시간차 타깃 계산
            next_v_value = agent.critic(tf.convert_to_tensor([next_state], dtype=tf.float32))
            v_values = agent.critic(tf.convert_to_tensor(states, dtype=tf.float32))
            gaes, y_i = agent.gae_target(rewards, v_values.numpy(), next_v_value.numpy(), done)

            # epoch만큼 반복
            for _ in range(20):
                print("===== update =====")
                # 액터 신경망 업데이트
                agent.actor_learn(tf.convert_to_tensor(log_old_policy_pdfs, dtype=tf.float32),
                                 tf.convert_to_tensor(states, dtype=tf.float32),
                                 tf.convert_to_tensor(actions, dtype=tf.float32),
                                 tf.convert_to_tensor(gaes, dtype=tf.float32))

                agent.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                  tf.convert_to_tensor(y_i, dtype=tf.float32))

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
