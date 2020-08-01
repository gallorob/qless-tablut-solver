import os

DEF = 0
ATK = 1

LAST_MOVES = [3, 5, 10, 20, 1000]
SHAPE_STATE = (9, 9, 3)

base_dir = os.path.dirname(__file__)

matches_dir = os.path.join(base_dir, 'matches')
os.makedirs(matches_dir, exist_ok=True)
datasets_dir = os.path.join(base_dir, 'datasets')
os.makedirs(datasets_dir, exist_ok=True)

records_dir = os.path.join(base_dir, 'test_recordings')
os.makedirs(records_dir, exist_ok=True)
