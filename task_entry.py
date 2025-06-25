try:
    from ..yaml_tools import load_environment_graph, load_objects_from_yaml, load_yaml
    from ... import REWARD_CLASSES
    from ...environments import BaseEnv, PsychopyEnv
except ImportError:
    from rewardgym.tasks.yaml_tools import load_environment_graph, load_objects_from_yaml, load_yaml
    from rewardgym import REWARD_CLASSES
    from rewardgym.environments import BaseEnv, PsychopyEnv

import numpy as np
from typing import Union, Literal
from pathlib import Path
from collections import defaultdict


task_dir = Path(__file__).parent


def get_task(render_backend=None, **kwargs):

    # All of these parts could also be defined as python dictionaries.
    environment_graph = load_environment_graph(task_dir / "graph.yaml")
    reward_structure = load_objects_from_yaml(task_dir / "rewards.yaml", REWARD_CLASSES)
    meta = load_yaml(task_dir / "meta.yaml")

    if render_backend is None:
        info_dict = defaultdict(int)
    else:
        info_dict = defaultdict(int)

    return environment_graph, reward_structure, info_dict, meta


def get_env(
    render_backend: Literal["pygame", "psychopy", "psychopy-simulate"] = None,
    seed: Union[int, np.random.Generator] = 1000,
    **kwargs,
):
    environment_graph, reward_structure, info_dict, meta = get_task(
        render_backend=render_backend,
        seed=seed)

    if render_backend is None:
        env = BaseEnv(
            environment_graph=environment_graph,
            reward_locations=reward_structure,
            render_mode=render_backend,
            info_dict=info_dict,
            seed=seed,
            name=meta['name'],
            reduced_actions=meta['reduced_actions'],
        )
    elif render_backend == "psychopy" or render_backend == "psychopy-simulate":
        env = PsychopyEnv(
            environment_graph=environment_graph,
            reward_locations=reward_structure,
            render_mode=render_backend,
            info_dict=info_dict,
            seed=seed,
            name=meta['name'],
            reduced_actions=meta['reduced_actions'],
        )

    return env