import gymnasium as gym

from . import agents
from .spot_curtain_env import SpotCurtainEnvCfg, SpotCurtainEnv

##
# Register Gym environments.
##

gym.register(
    id="MoDe-Spot-Curtain-v0",
    entry_point="task.Curtain.spot_curtain_env:SpotCurtainEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SpotCurtainEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml"
    },
)
