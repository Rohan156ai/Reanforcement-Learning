import gym
import numpy as np


a = gym.make('Blackjack-v0')


epsilon = 0.3
gamma = 0.9
Q = np.zeros((32,2))
alpha = 0.1




def choose(state):
    action = 0
    if epsilon > np.random.uniform():
        action = a.action_space.sample()
    else:
        action = np.argmax(Q[state])
    
    return action


def update(s1,a1,r,s2,a2):
    predict = Q[s1][a1]
    target = r + gamma*Q[s2][a2]
    Q[s1][a1] = Q[s1][a1] + alpha*(target - predict)

w = []
l = []

for i in range(1000):
    s1 = a.reset()
    s1 = np.array(s1)
    s1 = s1[0]
    
    for t in range(100):
        a1 = choose(s1)
        s2,r,d,_ = a.step(a1)
        s2 = np.array(s2)
        s2 = s2[0]
        a2 = choose(s2)
        update(s1,a1,r,s2,a2)
      
        s1 = s2
        a1 = a2
        
        if d:
            print(r)
            break
        

print(Q)


def play():
    t = 0
    s1 = a.reset()
    while t < 100:
        
        print(s1)
        s1 = np.array(s1)
        s1 = s1[0]
        a1 = np.argmax(Q[s1])
        print("action choose is ")
        print(a1)
        s2,r,done,_ = a.step(a1)
        print(s2)
        s1 = s2
        t = t + 1
        if done:
            if r == 1:
                print("Won")
            else:
                print("Loose")
            break
        
play()


