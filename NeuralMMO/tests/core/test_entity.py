import unittest
import numpy as np

import nmmo
from nmmo.entity.entity import Entity, EntityState
from nmmo.datastore.numpy_datastore import NumpyDatastore
from scripted.baselines import Random


class MockRealm:
  def __init__(self):
    self.config = nmmo.config.Default()
    self.config.PLAYERS = range(100)
    self.datastore = NumpyDatastore()
    self.datastore.register_object_type("Entity", EntityState.State.num_attributes)
    self._np_random = np.random

# pylint: disable=no-member
class TestEntity(unittest.TestCase):
  def test_entity(self):
    realm = MockRealm()
    entity_id = 123
    entity = Entity(realm, (10,20), entity_id, "name")

    self.assertEqual(entity.id.val, entity_id)
    self.assertEqual(entity.row.val, 10)
    self.assertEqual(entity.col.val, 20)
    self.assertEqual(entity.damage.val, 0)
    self.assertEqual(entity.time_alive.val, 0)
    self.assertEqual(entity.freeze.val, 0)
    self.assertEqual(entity.item_level.val, 0)
    self.assertEqual(entity.attacker_id.val, 0)
    self.assertEqual(entity.message.val, 0)
    self.assertEqual(entity.gold.val, 0)
    self.assertEqual(entity.health.val, realm.config.PLAYER_BASE_HEALTH)
    self.assertEqual(entity.food.val, realm.config.RESOURCE_BASE)
    self.assertEqual(entity.water.val, realm.config.RESOURCE_BASE)
    self.assertEqual(entity.melee_level.val, 0)
    self.assertEqual(entity.range_level.val, 0)
    self.assertEqual(entity.mage_level.val, 0)
    self.assertEqual(entity.fishing_level.val, 0)
    self.assertEqual(entity.herbalism_level.val, 0)
    self.assertEqual(entity.prospecting_level.val, 0)
    self.assertEqual(entity.carving_level.val, 0)
    self.assertEqual(entity.alchemy_level.val, 0)

  def test_query_by_ids(self):
    realm = MockRealm()
    entity_id = 123
    entity = Entity(realm, (10,20), entity_id, "name")

    entities = EntityState.Query.by_ids(realm.datastore, [entity_id])
    self.assertEqual(len(entities), 1)
    self.assertEqual(entities[0][Entity.State.attr_name_to_col["id"]], entity_id)
    self.assertEqual(entities[0][Entity.State.attr_name_to_col["row"]], 10)
    self.assertEqual(entities[0][Entity.State.attr_name_to_col["col"]], 20)

    entity.food.update(11)
    e_row = EntityState.Query.by_id(realm.datastore, entity_id)
    self.assertEqual(e_row[Entity.State.attr_name_to_col["food"]], 11)

  def test_recon_resurrect(self):
    config = nmmo.config.Default()
    config.set("PLAYERS", [Random])
    env = nmmo.Env(config)
    env.reset()

    # set player 1 to be a recon
    # Recons are immortal and cannot act (move)
    player1 = env.realm.players[1]
    player1.make_recon()
    spawn_pos = player1.pos

    for _ in range(50):  # long enough to starve to death
      env.step({})
      self.assertEqual(player1.pos, spawn_pos)
      self.assertEqual(player1.health.val, config.PLAYER_BASE_HEALTH)

    # resurrect player1
    player1.health.update(0)
    self.assertEqual(player1.alive, False)
    env.step({})

    player1.resurrect(health_prop=0.5, freeze_duration=10)
    self.assertEqual(player1.health.val, 50)
    self.assertEqual(player1.freeze.val, 10)
    self.assertEqual(player1.message.val, 0)
    self.assertEqual(player1.npc_type, -1)  # immortal flag
    self.assertEqual(player1.my_task.progress, 0)  # task progress should be reset
    # pylint:disable=protected-access
    self.assertEqual(player1._make_mortal_tick, env.realm.tick + 10)

if __name__ == '__main__':
  unittest.main()
