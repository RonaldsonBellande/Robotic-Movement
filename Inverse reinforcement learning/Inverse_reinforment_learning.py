import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

class IRL(object):

    def __init__(self):

        self.number_of_states = 25 #number of states excluding initial position
        self.trans_mat = np.zeros(((self.number_of_states + 1),4,(self.number_of_states + 1)))
        self.trans_mat = self.action_that_can_be_taken()
        self.state_features = np.eye((self.number_of_states + 1),self.number_of_states)  # Forcing zero reward at terminal state
        self.demos = [[0,1,2,3,4,9,14,19,24,25],[0,5,10,15,20,21,22,23,24,25],[0,5,6,11,12,17,18,23,24,25],[0,1,6,7,12,13,18,19,24,25]] # Demonstration
       ##For question 3
        self.trans_mat += -1
        self.trans_mat[25] = 10
       ############

        self.seed_weights = np.zeros(self.number_of_states)

        self.n_states = np.shape(self.trans_mat)[0]     # number of states
        self.n_actions = np.shape(self.trans_mat)[1]    # number of actions
        self.start_distance = np.zeros(np.shape(self.trans_mat)[0])
        self.state_freq = np.zeros(len(self.start_distance))
        self.n_features = np.shape(self.state_features)[1]
        self.policy = np.zeros((self.n_states,self.n_actions))
        self.r_weights = np.zeros(self.n_features)   #reward weight
        self.reward = np.matmul(self.state_features,self.r_weights)
        self.q_function = np.zeros((self.n_states,self.n_actions))
        self.value_function = np.zeros(self.n_states)


        self.start_distance[0] = 1
        self.n_epochs = 100
        self.horizon = 100
        self.learning_rate = 0.01

        self.dt = np.zeros((self.horizon,self.n_states))

        #########Question 3
        self.r_weight_question_3 = self.maxEntIRL()

        self.r_weights = self.maxEntIRL()
        self.reward_matrix = []

    def action_that_can_be_taken(self):
        
        for s in range(self.number_of_states): # Down Action
            if s < 20:
                self.trans_mat[s,0,s+5] = 1
            else:
                self.trans_mat[s,0,s] = 1

        for s in range(self.number_of_states): #Up Action
            if s >= 5:
                self.trans_mat[s,1,s-5] = 1
            else:
                self.trans_mat[s,1,s] = 1

        for s in range(self.number_of_states): # Left Action
            if s%5 > 0:
                self.trans_mat[s,2,s-1] = 1
            else:
                self.trans_mat[s,2,s] = 1

        for s in range(self.number_of_states): # Right Action
            if s%5 < 4:
                self.trans_mat[s,3,s+1] = 1
            else:
                self.trans_mat[s,3,s] = 1

        for a in range(4):
            self.trans_mat[(self.number_of_states - 1),a,self.number_of_states] = 1

        return self.trans_mat

    def calcMaxEntPolicy(self):

        for i in range(self.n_epochs):
            for s in range(self.number_of_states):
                for s_prime in range(self.number_of_states):
                    for a in range(self.n_actions):
                        if s!= (self.number_of_states + 1) and s_prime!=(self.number_of_states + 1):
                            self.q_function[s,a] = self.trans_mat[s,a,s_prime]*(self.reward[s]+((0.99)*self.value_function[s+1]))
                self.value_function[s] = max(self.q_function[s])

        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.policy[s,a] = np.argmax(self.q_function[s,a])
        return self.policy

    def calcExpectedStateFreq(self):
  
        for s in range(self.n_states):
            self.dt[0,s] = self.start_distance[s]
            for t in range(self.horizon-1):
                for a in range(self.n_actions):
                    for s_prime in range(self.n_states):
                        self.dt[t+1,s] += self.dt[t,s_prime]*self.policy[s_prime,a]*self.trans_mat[s_prime,a,s]
        self.state_freq = np.sum(self.dt,0)
        return self.state_freq

    def maxEntIRL(self):

        temp = 0
        for d in self.demos:
            for s in d:
                temp += self.state_features[s]
        f_lenght = temp/len(self.demos)

        for i in range(self.n_epochs):
            self.policy = self.calcMaxEntPolicy()
            svf = self.calcExpectedStateFreq()
            #question 5
            gradient = f_lenght - np.matmul(svf,self.state_features)
            self.r_weights = self.r_weights+(self.learning_rate*gradient)
        return self.r_weights

############Question 3
    def maxEntIRL_Question_3(self):
        self.start_distance[0] = 1
        f = 0
        for d in self.demos:
            for s in d:
                f += self.state_features[s]
        f_lenght = f/len(self.demos)

        for i in range(self.n_epochs):
            self.policy = self.calcMaxEntPolicy()
            print(self.policy)
            svf = self.calcExpectedStateFreq()
            self.r_weight_question_3 = self.r_weight_question_3+(self.learning_rate)
        return self.r_weight_question_3
########################

def main():

    iRL = IRL()
    for Si in range(iRL.number_of_states):
        iRL.reward_matrix.append( np.dot(iRL.r_weights, iRL.state_features[Si]) )
    iRL.reward_matrix = np.reshape(iRL.reward_matrix, (5,5))
    ##### question 3
    print(iRL.r_weight_question_3)
    print("second")
    #### Question 5
    print(iRL.r_weights)
    ##############
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    X = np.arange(0, 5, 1)
    Y = np.arange(0, 5, 1)
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, iRL.reward_matrix, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.show()

if __name__ == "__main__":
    main()