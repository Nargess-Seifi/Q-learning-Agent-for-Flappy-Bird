import utils
import flappy_bird_gym
import random
import time

class SmartFlappyBird:
    def __init__(self, iterations):
        self.Qvalues = utils.Counter()
        self.landa = 1 
        self.epsilon = 0.6 
        self.alpha = 0.86
        self.iterations = iterations

    @staticmethod
    def get_all_actions():
        return [0, 1]

    def convert_continuous_to_discrete(self, state):
        #20 bins each based on experiments
        x_discrete = int((state[0])*10)
        y_discrete = int((state[1] + 0.7) / (0.065))
        if x_discrete>20 or y_discrete>21:
            print("Something went wrong converting these values", f"state: {state}", f"Discretisized: {x_discrete} {y_discrete}")
        return (x_discrete, y_discrete)
    
    def compute_reward(self, prev_info, new_info, done, observation):
        if done:
            return -1000
        x_d, y_d = self.convert_continuous_to_discrete(observation)
        reward = -10 * abs(10 - y_d) *  (1 - ((x_d)/42)) #the closer we get to the next pipe the more severe the penalty will be
        reward += 1000 * (new_info['score'] - prev_info['score']) + 5
        return reward

    def policy(self, state):
        discrete_state = self.convert_continuous_to_discrete(state)
        return self.max_arg(discrete_state)

    def get_action(self, state): #get_action_for_updating
        if utils.flip_coin(self.epsilon):
            return random.choice(self.get_all_actions())
        else:
            return self.policy(state)

    def maxQ(self, state): 
        max_ = float('-inf')
        for action in self.get_all_actions():
            if self.Qvalues[(state, action)] > max_:
                max_ = self.Qvalues[(state, action)]
        return max_

    def max_arg(self, state):
          max_ = float('-inf')
          max_action = None
          for action in self.get_all_actions():
                if self.Qvalues[(state, action)] > max_:
                    max_ = self.Qvalues[(state, action)]
                    max_action = action
          return max_action
    
    def update(self, reward, state, action, next_state):
        self.Qvalues[(self.convert_continuous_to_discrete(state), action)] += self.alpha * (reward + self.landa * (self.maxQ(self.convert_continuous_to_discrete(next_state))) - self.Qvalues[(self.convert_continuous_to_discrete(state), action)])

    def update_epsilon_alpha(self, episode):
        self.epsilon = max(0.99999 * self.epsilon, 0.01)
        self.alpha = max(0.99999 * self.alpha, 0.01)

    def run_with_policy(self, landa):
        self.landa = landa
        env = flappy_bird_gym.make("FlappyBird-v0")
        for episode in range(self.iterations):
            observation = env.reset()
            info = {'score': 0}
            while True:
                action = self.get_action(observation)
                this_state = observation
                prev_info = info
                observation, reward, done, info = env.step(action)
                computed_reward = self.compute_reward(prev_info, info, done, observation)
                self.update(computed_reward, this_state, action, observation)
                self.update_epsilon_alpha(episode)
                if done:
                    break
            print(episode, "-", self.epsilon, self.alpha)
        env.close()

    def run_with_no_policy(self, landa):
        env = flappy_bird_gym.make("FlappyBird-v0")
        observation = env.reset()
        info = {'score': 0}
        while True:
            action = self.policy(observation)
            prev_info = info
            observation, reward, done, info = env.step(action)
            print(info)
            reward = self.compute_reward(prev_info, info, done, observation)
            print(reward)
            env.render()
            time.sleep(1 / 30)  # FPS
            if done:
                break
        env.close()
    def run(self):
        self.run_with_policy(0.9) 
        self.run_with_no_policy(0.9) 

program = SmartFlappyBird(iterations=50000)  
program.run()