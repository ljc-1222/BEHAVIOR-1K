from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES
from omnigibson.learning.wrappers.rgb_wrapper import RGBWrapper
from omnigibson.utils.ui_utils import create_module_logger


logger = create_module_logger("A2C2Wrapper")


class A2C2Wrapper(RGBWrapper):
    """
    Environment wrapper for online A2C2 evaluation.

    This keeps the RGBWrapper camera setup, adds only depth_linear for RGBD A2C2
    inputs, and adds the same task-info source used by replay_obs.py when it
    writes observation.task_info.
    """

    def __init__(self, env):
        super().__init__(env=env)
        robot = env.robots[0]
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            sensor_name = camera_name.split("::")[1]
            robot.sensors[sensor_name].add_modality("depth_linear")
        env.load_observation_space()
        self.env.task._include_obs = True
        logger.info("Enabled depth_linear and task low-dimensional observations for A2C2.")

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
