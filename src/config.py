from pathlib import Path

categories = ['neutral', 'happy', 'surprise']

_BASE_DIR = Path(__file__).resolve().parent
train_directory = str(_BASE_DIR / 'train')
test_directory = str(_BASE_DIR / 'test')

train_size = 5000
original_image_size = (48, 48)
image_size = (150, 150)
batch_size = 128
validation_split = 0.2

BOARD_SIZE = 3