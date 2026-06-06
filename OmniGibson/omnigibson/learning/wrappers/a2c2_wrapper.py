from omnigibson.learning.wrappers.default_wrapper import DefaultWrapper
from omnigibson.utils.ui_utils import create_module_logger


logger = create_module_logger("A2C2Wrapper")


class A2C2Wrapper(DefaultWrapper):
    """
    Environment wrapper for online A2C2 evaluation.

    This keeps DefaultWrapper's RGBD camera surface and adds the same task-info
    source used by replay_obs.py when it writes observation.task_info.
    """

    def __init__(self, env):
        super().__init__(env=env)
        self.env.task._include_obs = True
        logger.info("Enabled task low-dimensional observations for A2C2.")

    def _add_task_info(self, obs):
        task_obs = self.env.task.get_obs(self.env)
        if "low_dim" not in task_obs:
            raise KeyError("BEHAVIOR task observation did not include low_dim task_info.")
        obs["observation.task_info"] = task_obs["low_dim"]
        return obs

    def step(self, action, n_render_iterations=1):
        obs, reward, terminated, truncated, info = self.env.step(action, n_render_iterations=n_render_iterations)
        return self._add_task_info(obs), reward, terminated, truncated, info

    def reset(self):
        obs, info = self.env.reset()
        return self._add_task_info(obs), info
