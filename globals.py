import os
import shutil

DEF = 0
ATK = 1

LAST_MOVES = [5, 10, 20, 1000]
SHAPE_STATE = (9, 9, 3)

base_dir = os.path.dirname(__file__)

matches_dir = os.path.join(base_dir, 'matches')
os.makedirs(matches_dir, exist_ok=True)
datasets_dir = os.path.join(base_dir, 'datasets')
os.makedirs(datasets_dir, exist_ok=True)

records_dir = os.path.join(base_dir, 'test_recordings')
shutil.rmtree(records_dir)
os.makedirs(records_dir, exist_ok=True)
videos_dir = os.path.join(records_dir, 'videos')
os.makedirs(videos_dir, exist_ok=True)


class Settings:
    def __init__(self):
        self.TRAIN_VAL_SPLIT = 0.8
        self.TRAIN_TEST_SPLIT = 0.7
        self.N_MATCHES = 5
        self.EPOCHS = 5
        self.SIMULATE_MATCHES = False
        self.RENDER_SIMULATED_MATCHES = False
        self.GENERATE_DATASET = False
        self.TEST_MATCHES = 10
        self.RECORD_TEST_MATCHES = False
        self.TEST_MATCHES_RECORD_INTERVAL = 5
        self.RENDER_TEST_MATCHES = True
        self.USE_CUSTOM_GAMES = False


SETTINGS = Settings()

custom_games = {
    0: [
        # white won
        ['b5-b7', 'd5-d4', 'e2-c2', 'e4-h4', 'i6-h6', 'e5-e4', 'e8-g8', 'e3-e2', 'd1-d2', 'e4-e3', 'f1-f2', 'e3-a3'],
        ['e2-g2', 'e4-d4', 'e8-c8', 'd5-d6', 'h5-h7', 'e3-b3', 'd1-d3', 'c5-c2', 'g2-d2', 'e5-c5', 'a4-b4', 'e6-e2',
         'd3-d2', 'c5-c3', 'h7-h3', 'c2-c1', 'h3-e3', 'c1-d1', 'd2-c2', 'd1-c1', 'e1-e2', 'c3-c2', 'a5-a2', 'c1-d1',
         'a2-b2', 'c2-c1'],
        ['e2-g2', 'e4-d4', 'e8-c8', 'd5-d6', 'h5-h7', 'e3-b3', 'f1-f3', 'c5-c1', 'a4-b4', 'e5-c5', 'b5-b7', 'c5-c2',
         'a5-a2', 'c1-b1', 'd1-c1', 'd4-d1', 'a2-b2', 'c2-c1'],

        # black won
        ['h5-h7', 'f5-f4', 'b5-b3', 'd5-d6', 'e2-g2', 'c5-c9', 'a4-c4', 'd6-c6', 'd1-d3', 'e3-h3', 'b3-b9', 'e5-d5',
         'a5-b5', 'd5-d8', 'd3-d7'],
        ['h5-h3', 'e6-d6', 'e2-c2', 'f5-f8', 'f1-f7', 'g5-g7', 'e8-g8', 'e5-g5', 'h3-g3', 'e4-g4', 'i4-h4', 'd5-d4',
         'f9-f7', 'e3-e2', 'g3-e3', 'g4-g2', 'i6-g6', 'g5-f5', 'e1-f1', 'g2-g4', 'f1-f4', 'd4-b4', 'c2-c4', 'd6-d2',
         'e3-d3', 'e7-e3', 'f4-f3', 'f5-d5', 'a6-c6', 'd5-c5', 'i5-d5'],
        ['h5-h7', 'f5-f4', 'b5-b3', 'e4-d4', 'e8-c8', 'g5-g2', 'i6-g6', 'f4-h4', 'g6-g4', 'e5-g5', 'c8-g8', 'e3-g3',
         'h7-h3', 'g2-i2', 'i5-h5', 'g5-g4', 'h3-h4', 'd4-f4', 'e2-g2', 'e6-i6', 'f9-f6', 'd5-g5', 'h5-i5', 'g5-h5',
         'e9-i9', 'g3-h3', 'i9-i7', 'g4-g6', 'i7-i6', 'e7-e6', 'd1-d7', 'e6-e7', 'a6-c6', 'f4-f8', 'g2-g5', 'h3-h8',
         'd9-g9', 'g6-g7', 'i6-i7', 'e7-e9', 'c6-g6', 'f8-g8', 'e1-e7', 'i2-h2', 'i5-h5', 'h8-h7', 'f1-f9', 'h2-f2',
         'i4-f4', 'c5-c6', 'f4-f7', 'c6-e6', 'd7-d9', 'g8-e8', 'g9-g8']
    ]
}
