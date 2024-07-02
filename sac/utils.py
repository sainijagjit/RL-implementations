
import gymnasium

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update_and_freeze_gradient(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
        target_param.requires_grad = False
        
class MyWalkerWrapper(gymnasium.Wrapper):
    '''
    This is custom wrapper for BipedalWalker-v3 and BipedalWalkerHardcore-v3. 
    Rewards for failure is decreased to make agent brave for exploration and 
    time frequency of dynamic is lowered by skipping two frames.
    '''
    def __init__(self, env, skip=1):
        super().__init__(env)
        self._skip = skip
        self._max_episode_steps = 750
        
    def step(self, action):
        total_reward = 0
        obs, reward, done, trunc, info = self.env.step(action)
        for i in range(self._skip):
            if self.env.unwrapped.game_over:
                reward = -10.0
                info["dead"] = True
            else:
                info["dead"] = False
            total_reward += reward
            if done:
                break
        
        return obs, total_reward, done,trunc, info


    def render(self, mode="human"):
        for _ in range(self._skip):
            out = self.env.render(mode=mode)
        return out