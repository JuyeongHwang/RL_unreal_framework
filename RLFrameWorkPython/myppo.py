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

    def __init__(self,env):
        #hyperParams
        self.GAMMA = 0.90
        self.GAE_LAMBDA = 0.9
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.RATIO_CLIPPING = 0.2
        self.EPOCHS = 10

        self.env = env
        self.state_dim = env.observation_space.shape[0] #3차원 (x,y,각속도)
        self.action_dim = env.action_space.shape[0] #1차원, 토크
        print(f"dim {self.action_dim}")
        self.action_bound = 10 #env.action_space.high[0] #행동의 최대 크기. 2.0
        self.std_bound = [1e-2, 1.0] #표준편차 standard derivation의 최솟값, 최댓값 설정

        #액터 신경망 및 크리틱 신경망 생성.
        self.actor = Actor(self.action_dim, self.action_bound)
        self.actor.build(input_shape=(None,self.state_dim))
        #self.actor.summary()
        self.critic = Critic()
        self.critic.build(input_shape=(None,self.state_dim))
        #self.critic.summary()

        #옵티마이저
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(self.CRITIC_LEARNING_RATE)
        #에피소드로부터 얻은 총 보상값 저장
        self.save_epi_reward = []

    def log_pdf(self,mu,std,action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std**2
        log_policy_pdf = -0.5*(action-mu)**2/var-0.5*tf.math.log(var*2*np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def get_policy_action(self,state):
        mu_a,std_a = self.actor(state)

        mu_a = mu_a.numpy()[0]
        std_a = std_a.numpy()[0]
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size = self.action_dim)

        #print("상태 : ",state)
        #print("평균 : ", mu_a)
        #print("표준편차 : ",std_a)
        #print("행동 : ",action)

        return mu_a, std_a, action

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

    def unpack_batch(self,batch):
        unpack = batch[0]
        for idx in range(len(batch) - 1):
            unpack = np.append(unpack, batch[idx + 1], axis=0)

        return unpack

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

    def load_weights(self,path):
        self.actor.load_weights(path+'pendulum_actor.h5')
        self.critic.load_weights(path+'pendulum_critic.h5')

    def train(self, max_episode_num):

        #배치 초기화
        batch_state, batch_action,batch_reward = [],[],[]
        batch_log_old_policy_pdf = []

        #에피소드마다 다음을 반복
        for ep in range(int(max_episode_num)):

            #에피소드 초기화
            time,episode_reward,done = 0,0,False
            state = self.env.reset()
            state = state[0]

            while not done:
                mu_old, std_old, action = self.get_policy_action(tf.convert_to_tensor([state], dtype=tf.float32))
                print(f"행동 : {action}")
                action = np.clip(action, -self.action_bound, self.action_bound)

                #이전 정책의 로그 확률밀도함수 계산
                var_old = std_old**2
                log_old_policy_pdf = -0.5*(action-mu_old)**2/var_old - 0.5*np.log(var_old*2*np.pi)
                log_old_policy_pdf = np.sum(log_old_policy_pdf)

                '''
                print(" ")
                print("episodes ",ep)
                print("평균 : ", mu_old)
                print("표준편차 : ",std_old)
                print("행동 : ",action)
                print("이전 정책 로그 확률밀도함수 : ",log_old_policy_pdf)
                '''
                #다음 상태, 보상 관측

                next_state, reward,done,_,_ = self.env.step(action)
                print(reward)
                # shape 변환
                state = np.reshape(state,[1,self.state_dim])
                action = np.reshape(action,[1, self.action_dim])
                reward = np.reshape(reward,[1,1])

                log_old_policy_pdf = np.reshape(log_old_policy_pdf, [1,1])
                '''
                print("상태 : ",state)
                print("행동 : ",action)
                print("보상 : ",reward)
                print("이전 정책 로그 확률밀도함수  : ",log_old_policy_pdf)
                '''
                #학습용 보상 설정 (중심 옮기기)
                train_reward = (reward+8)/8

                #배치에 저장
                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append(reward)
                batch_log_old_policy_pdf.append(log_old_policy_pdf)

                #배치가 채워질때까지 학습하지 않고 저장만 계속함

                if (len(batch_state) < self.BATCH_SIZE):
                    #상태 업데이트
                    state = next_state
                    episode_reward += reward[0]
                    time+=1
                    continue

                #배치가 채워지면 학습진행
                #배치에서 데이터 추출
                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                rewards = self.unpack_batch(batch_reward)
                log_old_policy_pdfs = self.unpack_batch(batch_log_old_policy_pdf)
                #배치 비움
                batch_state,batch_action,batch_reward=[],[],[]
                batch_log_old_policy_pdf = []

                #GAE와 시간차 타깃 계산
                next_v_value = self.critic(tf.convert_to_tensor([next_state],dtype = tf.float32))
                v_values = self.critic(tf.convert_to_tensor(states, dtype=tf.float32))
                gaes,y_i = self.gae_target(rewards,v_values.numpy(), next_v_value.numpy(),done)

                #epoch만큼 반복
                for _ in range(self.EPOCHS):
                    #액터 신경망 업데이트
                    self.actor_learn(tf.convert_to_tensor(log_old_policy_pdfs, dtype=tf.float32),
                                     tf.convert_to_tensor(states,dtype = tf.float32),
                                     tf.convert_to_tensor(actions,dtype = tf.float32),
                                     tf.convert_to_tensor(gaes,dtype = tf.float32))

                    self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))

                #다음 에피소드를 위한 준비
                state = next_state
                episode_reward += reward[0]
                time +=1

                bFirstState= False
                done = True

            print('Episode: ',ep+1,'Time: ',time,'Reward: ',episode_reward)
            self.save_epi_reward.append(episode_reward)

            if ep%10==0:
                self.actor.save_weights("./save_weights/pendulum_actor.h5")
                self.critic.save_weights("./save_weights/pendulum_critic.h5")

        np.savetxt('./save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)

    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()



def main():
    max_episode_num = 1
    env = gym.make('Pendulum-v1', render_mode = 'human')

    agent = PPOAgent(env)
    agent.train(max_episode_num)
    agent.plot_result()

if __name__=="__main__":
    main()


'''
env.reset()


env.reset()
env.render()
state = env.step(env.action_space.sample())[0]

state = env.step(env.action_space.sample())[0]

mu_a, std_a, action = agent.get_policy_action(tf.convert_to_tensor([state], dtype=tf.float32))

agent.actor_learn(1, tf.convert_to_tensor([state], dtype=tf.float32), action,0)
agent.critic_learn(tf.convert_to_tensor([state], dtype=tf.float32), 10)
'''