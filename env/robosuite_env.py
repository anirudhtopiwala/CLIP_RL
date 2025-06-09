from robosuite import make
import numpy as np
from gym import Wrapper


class DenseRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.prev_gripper_to_cube_dist = None

    def reset(self):
        obs = self.env.reset()
        self.prev_gripper_to_cube_dist = self._get_gripper_to_cube_distance()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Original sparse reward
        sparse_reward = reward

        # Get current positions
        cube_pos = self._get_cube_pos()
        gripper_to_cube_dist = self._get_gripper_to_cube_distance()

        # Dense reward components
        # 1. Reward for moving closer to cube
        distance_reward = 0
        if self.prev_gripper_to_cube_dist is not None:
            distance_reward = self.prev_gripper_to_cube_dist - gripper_to_cube_dist

        # 2. Reward for cube height (lifting)
        height_reward = max(0, cube_pos[2] - 0.8)  # 0.8 is assumed table height

        # 3. Proximity reward
        proximity_reward = 0
        if gripper_to_cube_dist < 0.1:
            proximity_reward = 0.1 - gripper_to_cube_dist

        # Update previous values
        self.prev_gripper_to_cube_dist = gripper_to_cube_dist

        # Normalize and scale rewards with explicit weights.
        distance_coef = 1
        height_coef = 2
        proximity_coef = 1.5

        # Combine rewards (original + normalized dense shaping)
        total_reward = (
            sparse_reward
            + distance_coef * distance_reward
            + height_coef * height_reward
            + proximity_coef * proximity_reward
        )

        return obs, total_reward, done, info

    def _get_gripper_to_cube_distance(self):
        obs = self.env._get_observations()
        gripper_pos = obs["robot0_eef_pos"]
        cube_pos = obs["cube_pos"]
        return np.linalg.norm(gripper_pos - cube_pos)

    def _get_cube_pos(self):
        obs = self.env._get_observations()
        return obs["cube_pos"]


def create_env(task="Lift", use_camera=True, is_renderer=True):
    env = make(
        env_name=task,
        robots="Panda",
        has_renderer=is_renderer,
        has_offscreen_renderer=use_camera,
        use_camera_obs=use_camera,
        use_object_obs=True,
        camera_names="frontview",
        reward_shaping=False,
        horizon=512,
        camera_heights=84,
        camera_widths=84,
    )
    return DenseRewardWrapper(env)
