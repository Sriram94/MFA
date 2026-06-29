from ijcai2022nmmo.env.team_based_env import TeamBasedEnv
from ijcai2022nmmo.evaluation.team import Team
from ijcai2022nmmo.evaluation.rollout import RollOut
from ijcai2022nmmo.config import CompetitionConfig
from ijcai2022nmmo.evaluation.proxy import ProxyTeam
from ijcai2022nmmo.evaluation.proxy import TeamServer
from ijcai2022nmmo.env.metrics import Metrics
from ijcai2022nmmo.env.stat import Stat
from ijcai2022nmmo.timer import timer
from ijcai2022nmmo.evaluation.rating import RatingSystem
from ijcai2022nmmo.evaluation.analyzer import TeamResult
from ijcai2022nmmo.evaluation import analyzer
from ijcai2022nmmo import exception

__all__ = [
    "TeamBasedEnv",
    "Team",
    "RollOut",
    "CompetitionConfig",
    "ProxyTeam",
    "TeamServer",
    "Metrics",
    "Stat",
    "timer",
    "RatingSystem",
    "TeamResult",
    "analyzer",
    "exception",
]

from ijcai2022nmmo.version import version

__version__ = version

# monkey patch nmmo
from nmmo.entity import Player


def _packet(self):
    data = super(Player, self).packet()

    data["entID"] = self.entID
    data["annID"] = self.population

    data["base"] = self.base.packet()
    data["resource"] = self.resources.packet()
    data["skills"] = self.skills.packet()

    data["metrics"] = {
        "PlayerDefeats": self.history.playerKills,
        "Equipment": self.loadout.defense,
        "Exploration": self.history.exploration,
        "Foraging":
        (self.skills.fishing.level + self.skills.hunting.level) / 2.0,
        "Achievement": self.diary.cumulative_reward,
        "TimeAlive": self.history.timeAlive.val,
    }

    return data


def _monkey_patch():
    Player.packet = _packet


_monkey_patch()
