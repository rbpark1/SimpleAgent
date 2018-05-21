from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

# constants
_NOOP = actions.FUNCTIONS.no_op.id
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_TERRAN_MARINE = 48
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_NOT_QUEUED = [0]

_SELECT_ARMY = actions.FUNCTIONS.select_army.id


class SimpleAgent(base_agent.BaseAgent):


    def step(self, obs):
        super(SimpleAgent, self).step(obs)

        if _SELECT_ARMY in obs.observation["available_actions"]:
            self.army_selected = True
            return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        return actions.FunctionCall(_NOOP, [])