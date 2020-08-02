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
videos_dir = os.path.join(records_dir, 'videos')
os.makedirs(videos_dir, exist_ok=True)


class Settings:
    def __init__(self):
        self.TRAIN_VAL_SPLIT = 0.8
        self.TRAIN_TEST_SPLIT = 0.7
        self.N_MATCHES = 10
        self.EPOCHS = 5
        self.SIMULATE_MATCHES = True
        self.GENERATE_DATASET = True
        self.TEST_MATCHES = 5
        self.RECORD_TEST_MATCHES = False
        self.TEST_MATCHES_RECORD_INTERVAL = 5
        self.RENDER_TEST_MATCHES = False


SETTINGS = Settings()
