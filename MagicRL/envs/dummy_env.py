from MagicRL.envs.base_env import BaseEnv


class DummyEnv(BaseEnv):
    def sample_state(self, args: dict = None):
        raise NotImplementedError

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = None

    def reset(self):
        self.state = 0

    def step(self, action):
        self.state += 1
        return self.state, 0, self.state < 10, None
