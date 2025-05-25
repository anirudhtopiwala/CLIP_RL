from robosuite import make


def create_env(task="Lift", use_camera=True, is_renderer=True):
    env = make(
        env_name=task,
        robots="Panda",
        has_renderer=is_renderer,
        has_offscreen_renderer=use_camera,
        use_camera_obs=use_camera,
        camera_names="frontview",
        reward_shaping=True,
        horizon=250,
        camera_heights=84,
        camera_widths=84,
    )
    return env
