from torch.distributions import Categorical
import torch
import gym
from torch import optim
from tqdm import tqdm
import numpy as np
class Displayer:
    def __init__(self,model,env) -> None:
        self.model=model
        self.env=env

    def display(self) -> None:
        self.model.eval()
        state=self.env.reset()
        while True:
            self.env.render()#窗口动画，不适合笔记本环境
            action_prob = self.model(torch.FloatTensor(state))
            action_dist = Categorical(action_prob)
            action = action_dist.sample().item()
            state, reward, done, _ = self.env.step(action)
            if done :
                self.env.render()
                break

class PolicyGradientAgent():
    
    def __init__(self, network,lr) -> None:
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=lr)

    def learn(self, log_probs, rewards) -> None:
        loss = (-log_probs * rewards).sum()#因为SGD做的是梯度下降，而我们理论上是梯度上升，所以加负号

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

class PolicyGradientTrainer:
    def __init__(self,epochs,batch_size,agent:PolicyGradientAgent,env) -> None:
        self.agent=agent
        self.epochs=epochs
        self.batch_size=batch_size
        self.env=env

    def train(self) -> None:
        self.agent.network.train()
        avg_total_rewards, avg_final_rewards = [], []
        prg_bar = tqdm(range(self.epochs))
        for epoch in prg_bar:
            log_probs, rewards = [], []
            total_rewards, final_rewards = [], []
            for episode in range(self.batch_size):
                state = self.env.reset()
                total_reward, total_step = 0, 0
                while True:
            
                    action, log_prob = self.agent.sample(state)
                    next_state, reward, done, _ = self.env.step(action)

                    log_probs.append(log_prob)
                    state = next_state
                    total_reward += reward
                    total_step += 1

                    if done:
                        final_rewards.append(reward)
                        total_rewards.append(total_reward)
                        rewards.append(np.full(total_step, total_reward))  
                        break
            avg_total_reward = sum(total_rewards) / len(total_rewards)
            avg_final_reward = sum(final_rewards) / len(final_rewards)
            avg_total_rewards.append(avg_total_reward)
            avg_final_rewards.append(avg_final_reward)
            prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

            rewards = np.concatenate(rewards, axis=0)
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9) #主要是前面那一步，减去一个baseline，除以标准差是防止走太快 
            self.agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))

class PolicyGradientLearner:

    def __init__(self,args,model_class,model_path=None) -> None:
        self.args=args
        self.env=gym.make(self.args.game)
        self.model=model_class()
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        self.agent=PolicyGradientAgent(self.model,self.args.lr)
        self.trainer=PolicyGradientTrainer(
            self.args.epochs,
            self.args.batch_size,
            self.agent,
            self.env
        )
        self.displayer=Displayer(self.model,self.env)

    def train(self) -> None:
        self.trainer.train()

    def display(self) -> None:
        self.displayer.display()
        
    def save_model(self,path) -> None:
        torch.save(self.model.state_dict(),path)