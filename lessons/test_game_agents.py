"""Behavior checks for the generated, trainable game examples (requires PyTorch)."""
from pathlib import Path
import runpy
import unittest

ROOT = Path(__file__).resolve().parent.parent


def load_example(name):
    return runpy.run_path(str(ROOT/'examples/python/cpu'/f'{name}.py'))


class GameAgentTests(unittest.TestCase):
    def test_rectangular_grids_learn_shortest_paths_from_every_start(self):
        example = load_example('lab-game-grid')
        for rows, cols in [(4, 5), (3, 7), (7, 2), (1, 5)]:
            with self.subTest(rows=rows, cols=cols):
                world = example['make_world'](rows, cols)
                policy, loss = example['train'](world)
                self.assertLess(loss, 1e-4)
                for r in range(rows):
                    for c in range(cols):
                        if (r, c) == world['goal']:
                            continue
                        result = example['rollout'](world, lambda state: policy[state], start=(r, c))
                        self.assertTrue(result['done'])
                        self.assertEqual(len(result['actions']), rows - 1 - r + cols - 1 - c)

    def test_item_values_change_the_learned_route_and_match_the_optimal_return(self):
        example = load_example('lab-game-items')
        routes = []
        for coin_value in [1.0, 20.0]:
            with self.subTest(coin_value=coin_value):
                world = example['make_world'](4, 5, walls={(1, 1), (1, 2), (2, 3)}, items={
                    (0, 4): ('Монета', coin_value), (2, 1): ('Алмаз', 5.0), (3, 0): ('Сундук', 9.0)})
                policy, _ = example['train'](world)
                result = example['rollout'](world, lambda state: policy[state])
                self.assertTrue(result['done'])
                self.assertTrue(all(s[:2] not in world['walls'] for s in result['route']))
                # Independent Bellman value iteration checks the learned policy's return.
                values = {state: 0.0 for state in policy}
                for _ in range(400):
                    next_values = {}
                    for state in values:
                        next_values[state] = max(reward + (0.0 if done else 0.95 * values[nxt])
                            for nxt, reward, done in (example['step'](world, state, a) for a in range(4)))
                    difference = max(abs(next_values[s] - values[s]) for s in values)
                    values = next_values
                    if difference < 1e-9:
                        break
                self.assertAlmostEqual(result['discounted'], values[(0, 0, 0)], delta=0.02)
                routes.append(result['route'])
                first_collected = next(s[2] for s in result['route'] if s[2])
                self.assertEqual(first_collected, 4 if coin_value == 1.0 else 1)
        self.assertNotEqual(routes[0], routes[1])
        self.assertEqual(set(example['rollout'](world, lambda state: policy[state])['actions']), {0, 1, 2, 3})


if __name__ == '__main__':
    unittest.main()
