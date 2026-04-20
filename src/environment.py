import gymnasium as gym
import numpy as np
import vizdoom as zd
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecVideoRecorder
import os

class DoomEnv(gym.Env):
    def __init__(self, game, render_mode="rgb_array"):
        super(DoomEnv, self).__init__()
        self.game = game
        self.render_mode = render_mode 
        
        self.action_space = gym.spaces.Discrete(self.game.get_available_buttons_size())
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)

    def step(self, action):
        actions = [0] * self.game.get_available_buttons_size()
        actions[action] = 1
        
        reward = self.game.make_action(actions)
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        
        if state:
            obs = self._preprocess(state.screen_buffer)
        else:
            obs = np.zeros((1, 84, 84), dtype=np.uint8)
            
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        state = self.game.get_state()
        return self._preprocess(state.screen_buffer), {}

    def render(self):
        state = self.game.get_state()
        if state:
            return state.screen_buffer 
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def _preprocess(self, img):
        import cv2
        img = cv2.resize(img, (84, 84))
        
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        return np.reshape(img, (1, 84, 84))

def make_doom_env(config, record=False):
    def _init():
        game = zd.DoomGame()
        cfg_path = os.path.normpath(os.path.join(config['SCENARIOS_DIR'], config['env_name']))
        game.load_config(cfg_path)
        
        mode = "rgb_array" if record else ("human" if config['render'] else "rgb_array")
        
        game.set_window_visible(config['render'])
        game.set_screen_format(zd.ScreenFormat.RGB24) 
        game.set_screen_resolution(zd.ScreenResolution.RES_320X240)
        game.init()
        
        return DoomEnv(game, render_mode=mode)

    env = DummyVecEnv([_init])
    env = VecFrameStack(env, n_stack=4)
    
    if record:
        os.makedirs(config['VIDEO_DIR'], exist_ok=True)
        env = VecVideoRecorder(env, config['VIDEO_DIR'], 
                               record_video_trigger=lambda x: x == 0, 
                               video_length=2000)
    return env