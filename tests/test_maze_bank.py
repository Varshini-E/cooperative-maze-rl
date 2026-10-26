import copy
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from PIL import ImageFont

from panel_guide.conversation import Conversation
from panel_guide.curriculum import task_id
from panel_guide.engine import MazeGame
from panel_guide.generate_mazes import write_previews
from panel_guide.manual import build_manual
from panel_guide.maze_bank import (SPLITS, generate_bank, load_bank, maze_id, select_tasks,
                                   topology_id, write_bank)
from panel_guide.render import render_maze
from panel_guide.solver import solve
from panel_guide.tasks import State, example_task, task_from_dict


class MazeBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.records = generate_bank(0, (1, 1, 1), 1)

    def test_generation_reproduces_exact_bank_and_seed_changes_tasks(self):
        manifest, records = generate_bank(0, (1, 1, 1), 1)
        self.assertEqual((manifest, records), (self.manifest, self.records))
        _, other = generate_bank(1, (1, 1, 1), 1)
        self.assertNotEqual({r['task_id'] for r in records}, {r['task_id'] for r in other})
        with self.assertRaises(ValueError):
            generate_bank(0, (1, 1, 1), 1, max_candidates=1)

    def test_all_generated_solutions_execute_and_visibility_stays_local(self):
        self.assertEqual(len(self.records), 60)
        self.assertEqual({row['difficulty'] for row in self.records}, {'easy', 'medium', 'hard'})
        for row in self.records:
            task = task_from_dict(row['task'])
            self.assertEqual(task_id(task), row['task_id'])
            game = MazeGame(task, row['solution_length'])
            route = solve(task)
            self.assertEqual(len(route), row['solution_length'])
            for action in route:
                result = game.step(action)
                self.assertIn(result.outcome, ('moved', 'toggled'))
            self.assertEqual((result.status, result.reward), ('success', 1))
            view = MazeGame(task).maze_view
            for room in view.rooms:
                if not room.current:
                    self.assertTrue(all(d.is_open is None and d.id is None for d in room.doors))
                    self.assertTrue(all(c.is_on is None and c.id is None for c in room.controls))
            # Manual is invariant to the reset state; it contains no oracle route or starting-state metadata.
            other_start = next(r.id for r in task.rooms if r.id not in (task.goal, task.initial.room))
            self.assertEqual(build_manual(task), build_manual(replace(task, initial=State(other_start, (True, True)))))
            request = Conversation(task, prompt_version='v4', observation_mode='maze_detail', operator_history='latest').start()
            self.assertIsNone(request.image_png)
            self.assertNotIn('solution_length', json.dumps(request.text_messages()))

    def test_structural_ids_ignore_room_order_names_symbols_and_geometry(self):
        task = example_task()
        names = {room.id: f'new-{index}' for index, room in enumerate(task.rooms)}
        emblems = [room.emblem for room in task.rooms][::-1]
        rooms = tuple(replace(room, id=names[room.id], emblem=emblems[index],
            position=(room.position[1], 3 - room.position[0]),
            doors=tuple(replace(door, destination=names[door.destination]) for door in room.doors))
            for index, room in enumerate(task.rooms))
        renamed = replace(task, rooms=rooms[::-1], goal=names[task.goal],
                          initial=State(names[task.initial.room], (True, True)))
        self.assertEqual(topology_id(task), topology_id(renamed))
        self.assertEqual(maze_id(task), maze_id(renamed))
        self.assertNotEqual(task_id(task), task_id(renamed))
        different_goal = replace(task, goal='Workshop')
        self.assertEqual(topology_id(task), topology_id(different_goal))
        self.assertNotEqual(maze_id(task), maze_id(different_goal))

    def test_topology_and_entire_start_room_partitions_are_disjoint(self):
        families, starts, resets = defaultdict(set), defaultdict(lambda: defaultdict(set)), defaultdict(set)
        for row in self.records:
            partition = 'train' if row['split'] in ('train', 'validation_starts', 'test_starts') else row['split']
            families[partition].add(row['topology_id'])
            room = row['task']['initial']['room']
            starts[row['maze_id']][row['split']].add(room)
            resets[row['maze_id'], room].add(tuple(row['task']['initial']['switches']))
        keys = list(families)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                self.assertTrue(families[a].isdisjoint(families[b]))
        for parts in starts.values():
            seen = set()
            for rooms in parts.values():
                self.assertTrue(seen.isdisjoint(rooms))
                seen |= rooms
        self.assertTrue(all(len(bits) == 4 for bits in resets.values()))

    def test_loading_checks_checksums_metadata_and_split_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_bank(root, self.manifest, self.records)
            _, loaded = load_bank(root)
            self.assertEqual(json.loads(json.dumps(self.records)), loaded)
            write_previews(root, loaded)
            self.assertEqual(len(list((root / 'previews').glob('*.png'))), 2)
            with self.assertRaises(ValueError):
                write_bank(root, self.manifest, self.records)
            with (root / 'tasks.jsonl').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_bank(root)
        for fault in ('start_room', 'topology', 'difficulty'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                manifest, rows = copy.deepcopy(self.manifest), copy.deepcopy(self.records)
                row = next(r for r in rows if r['split'] == 'train')
                if fault == 'start_room':
                    row['split'] = 'test_starts'
                elif fault == 'topology':
                    row['topology_id'] = 'renamed-topology'
                else:
                    row['solution_length'] += 1
                write_bank(Path(folder), manifest, rows)  # Fresh checksum must not bypass semantic validation.
                with self.assertRaises(ValueError):
                    load_bank(Path(folder))

    def test_selection_is_balanced_reproducible_and_never_crosses_splits(self):
        selected = select_tasks(self.records, 'train', 9, 123)
        self.assertEqual(Counter(r['difficulty'] for r in selected), {'easy': 3, 'medium': 3, 'hard': 3})
        self.assertEqual(selected, select_tasks(self.records, 'train', 9, 123))
        self.assertEqual(len({r['task_id'] for r in selected}), 9)
        for split in SPLITS:
            band = next(r['difficulty'] for r in self.records if r['split'] == split)
            self.assertTrue(all(r['split'] == split for r in select_tasks(self.records, split, 2, 0, (band,))))
        with self.assertRaises(ValueError):
            select_tasks(self.records, 'train', 100)
        with self.assertRaises(ValueError):
            select_tasks(self.records, 'test_starts', 1, difficulties=('hard',))

    def test_generated_narrow_floorplans_leave_space_for_the_legend(self):
        checked = set()
        for row in self.records:
            if row['topology_id'] in checked:
                continue
            checked.add(row['topology_id'])
            view = MazeGame(task_from_dict(row['task'])).maze_view
            with render_maze(view) as image:
                left_end = 48 + 309 + ImageFont.load_default(size=19).getlength('State hidden until you enter the room')
                right_start = image.width - 48 - ImageFont.load_default(size=16).getlength('Only local IDs can be used for actions')
                self.assertLess(left_end, right_start)


if __name__ == '__main__':
    unittest.main()
