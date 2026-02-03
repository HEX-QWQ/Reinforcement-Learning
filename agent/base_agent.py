class BaseAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, obs, deterministic=False):
        raise NotImplementedError
