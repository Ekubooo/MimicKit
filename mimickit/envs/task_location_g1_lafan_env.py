"""Original Location task with G1 observations adapted to the frozen LaFAN prior."""
from envs.g1_lafan_env import G1LafanEnvMixin
from envs.task_location_env import TaskLocationEnv


class TaskLocationG1LafanEnv(G1LafanEnvMixin, TaskLocationEnv):
    pass
