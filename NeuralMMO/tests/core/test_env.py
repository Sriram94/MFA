import unittest
from typing import List

import random
import numpy as np
from tqdm import tqdm

import nmmo
from nmmo.core.realm import Realm
from nmmo.core.tile import TileState
from nmmo.entity.entity import Entity, EntityState
from nmmo.systems.item import ItemState
from scripted import baselines

# Allow private access for testing
# pylint: disable=protected-access

# 30 seems to be enough to test variety of agent actions
TEST_HORIZON = 30
RANDOM_SEED = random.randint(0, 10000)

class Config(nmmo.config.Small, nmmo.config.AllGameSystems):
  PLAYERS = [
    baselines.Fisher, baselines.Herbalist, baselines.Prospector,
    baselines.Carver, baselines.Alchemist,
    baselines.Melee, baselines.Range, baselines.Mage]

class TestEnv(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.config = Config()
    cls.env = nmmo.Env(cls.config, RANDOM_SEED)

  def test_action_space(self):
    action_space = self.env.action_space(0)
    atn_str_keys = set(atn.__name__ for atn in nmmo.Action.edges(self.config))
    self.assertSetEqual(
        set(action_space.keys()),
        atn_str_keys)

  def test_observations(self):
    obs, _ = self.env.reset()
    self.assertEqual(obs.keys(), self.env.realm.players.keys())

    for _ in tqdm(range(TEST_HORIZON)):
      entity_locations = [
        [ev.row.val, ev.col.val, e] for e, ev in self.env.realm.players.entities.items()
      ] + [
        [ev.row.val, ev.col.val, e] for e, ev in self.env.realm.npcs.entities.items()
      ]

      for player_id, player_obs in obs.items():
        if player_id in self.env.realm.players: # alive agents
          self._validate_tiles(player_obs, self.env.realm)
          self._validate_entitites(
              player_id, player_obs, self.env.realm, entity_locations)
          self._validate_inventory(player_id, player_obs, self.env.realm)
          self._validate_market(player_obs, self.env.realm)
        else:
          # the obs of dead agents are dummy, all zeros
          self.assertEqual(np.sum(player_obs["Tile"]), 0)
          self.assertEqual(np.sum(player_obs["Entity"]), 0)
          self.assertEqual(np.sum(player_obs["Inventory"]), 0)
          self.assertEqual(np.sum(player_obs["Market"]), 0)
          self.assertEqual(np.sum(player_obs["ActionTargets"]["Move"]["Direction"]), 1)  # no-op
          self.assertEqual(np.sum(player_obs["ActionTargets"]["Attack"]["Style"]), 3)  # all ones

      obs, rewards, terminated, truncated, infos = self.env.step({})

      # make sure dead agents return proper dones=True, dummy obs, and -1 reward
      self.assertEqual(len(self.env.agents), len(self.env.realm.players))
      # NOTE: the below is no longer true when mini games resurrect dead players
      # self.assertEqual(len(self.env.possible_agents),
      #                  len(self.env.realm.players) + len(self.env._dead_agents))
      for agent_id in self.env.agents:
        self.assertTrue(agent_id in obs)
        self.assertTrue(agent_id in rewards)
        self.assertTrue(agent_id in terminated)
        self.assertTrue(agent_id in truncated)
        self.assertTrue(agent_id in infos)
      for dead_id in self.env._dead_this_tick:
        self.assertEqual(rewards[dead_id], -1)
        self.assertTrue(terminated[dead_id])

      # check dead and alive
      entity_all = EntityState.Query.table(self.env.realm.datastore)
      alive_agents = entity_all[:, Entity.State.attr_name_to_col["id"]]
      alive_agents = set(alive_agents[alive_agents > 0])
      for agent_id in alive_agents:
        self.assertTrue(agent_id in self.env.realm.players)

  def _validate_tiles(self, obs, realm: Realm):
    for tile_obs in obs["Tile"]:
      tile_obs = TileState.parse_array(tile_obs)
      tile = realm.map.tiles[(int(tile_obs.row), int(tile_obs.col))]
      for key, val in tile_obs.__dict__.items():
        if val != getattr(tile, key).val:
          self.assertEqual(val, getattr(tile, key).val,
            f"Mismatch for {key} in tile {tile_obs.row}, {tile_obs.col}")

  def _validate_entitites(self, player_id, obs, realm: Realm, entity_locations: List[List[int]]):
    observed_entities = set()

    for entity_obs in obs["Entity"]:
      entity_obs = EntityState.parse_array(entity_obs)

      if entity_obs.id == 0:
        continue

      entity: Entity = realm.entity(entity_obs.id)

      observed_entities.add(entity.ent_id)

      for key, val in entity_obs.__dict__.items():
        if getattr(entity, key) is None:
          raise ValueError(f"Entity {entity} has no attribute {key}")
        self.assertEqual(val, getattr(entity, key).val,
          f"Mismatch for {key} in entity {entity_obs.id}")

    # Make sure that we see entities IFF they are in our vision radius
    row = realm.players.entities[player_id].row.val
    col = realm.players.entities[player_id].col.val
    vision = realm.config.PLAYER_VISION_RADIUS
    visible_entities = {
      e for r, c, e in entity_locations
      if row - vision <= r <= row + vision and col - vision <= c <= col + vision
    }
    self.assertSetEqual(visible_entities, observed_entities,
      f"Mismatch between observed: {observed_entities} " \
        f"and visible {visible_entities} for player {player_id}, "\
        f" step {self.env.realm.tick}")

  def _validate_inventory(self, player_id, obs, realm: Realm):
    self._validate_items(
        {i.id.val: i for i in realm.players[player_id].inventory.items},
        obs["Inventory"]
    )

  def _validate_market(self, obs, realm: Realm):
    self._validate_items(
        {i.item.id.val: i.item for i in realm.exchange._item_listings.values()},
        obs["Market"]
    )

  def _validate_items(self, items_dict, item_obs):
    item_obs = item_obs[item_obs[:,0] != 0]
    if len(items_dict) != len(item_obs):
      assert len(items_dict) == len(item_obs),\
        f"Mismatch in number of items. Random seed: {RANDOM_SEED}"
    for item_ob in item_obs:
      item_ob = ItemState.parse_array(item_ob)
      item = items_dict[item_ob.id]
      for key, val in item_ob.__dict__.items():
        self.assertEqual(val, getattr(item, key).val,
          f"Mismatch for {key} in item {item_ob.id}: {val} != {getattr(item, key).val}")

  def test_clean_item_after_reset(self):
    # use the separate env
    new_env = nmmo.Env(self.config, RANDOM_SEED)

    # reset the environment after running
    new_env.reset()
    for _ in tqdm(range(TEST_HORIZON)):
      new_env.step({})
    new_env.reset()

    # items are referenced in the realm.items, which must be empty
    self.assertTrue(len(new_env.realm.items) == 0)
    self.assertTrue(len(new_env.realm.exchange._item_listings) == 0)
    self.assertTrue(len(new_env.realm.exchange._listings_queue) == 0)

    # item state table must be empty after reset
    self.assertTrue(ItemState.State.table(new_env.realm.datastore).is_empty())

  def test_truncated(self):
    test_horizon = 25
    config = Config()
    config.set("HORIZON", test_horizon)
    env = nmmo.Env(config, RANDOM_SEED)
    obs, _ = env.reset()
    for _ in tqdm(range(test_horizon)):
      obs, _, terminated, truncated, _ = env.step({})
      for agent_id in obs:
        alive = agent_id in env.realm.players
        self.assertEqual(terminated[agent_id], not alive)
        # Test that the last step is truncated
        self.assertEqual(truncated[agent_id], alive and env.realm.tick >= test_horizon)

if __name__ == '__main__':
  unittest.main()
