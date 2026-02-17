from config import BOARD_SIZE, categories, image_size
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

class TicTacToePlayer:
    def get_move(self, board_state):
        raise NotImplementedError()

class UserInputPlayer:
    def get_move(self, board_state):
        inp = input('Enter x y:')
        try:
            x, y = inp.split()
            x, y = int(x), int(y)
            return x, y
        except Exception:
            return None

import random

class RandomPlayer:
    def get_move(self, board_state):
        positions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board_state[i][j] is None:
                    positions.append((i, j))
        return random.choice(positions)

from matplotlib import pyplot as plt
from matplotlib.image import imread
import cv2

class UserWebcamPlayer:
    def _process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width, height = frame.shape
        size = min(width, height)
        pad = int((width-size)/2), int((height-size)/2)
        frame = frame[pad[0]:pad[0]+size, pad[1]:pad[1]+size]
        return frame

    def _access_webcam(self):
        import cv2
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)
        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
            frame = self._process_frame(frame)
        else:
            rval = False
        while rval:
            cv2.imshow("preview", frame)
            rval, frame = vc.read()
            frame = self._process_frame(frame)
            key = cv2.waitKey(20)
            if key == 13: # exit on Enter
                break

        vc.release()
        cv2.destroyWindow("preview")
        return frame

    def _print_reference(self, row_or_col):
        print('reference:')
        for i, emotion in enumerate(categories):
            print('{} {} is {}.'.format(row_or_col, i, emotion))
    
    def _get_row_or_col_by_text(self):
        try:
            val = int(input())
            return val
        except Exception as e:
            print('Invalid position')
            return None
    
    def _get_row_or_col(self, is_row):
        try:
            row_or_col = 'row' if is_row else 'col'
            self._print_reference(row_or_col)
            img = self._access_webcam()
            emotion = self._get_emotion(img)
            if type(emotion) is not int or emotion not in range(len(categories)):
                print('Invalid emotion number {}'.format(emotion))
                return None
            print('Emotion detected as {} ({} {}). Enter \'text\' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.'.format(categories[emotion], row_or_col, emotion))
            inp = input()
            if inp == 'text':
                return self._get_row_or_col_by_text()
            return emotion
        except Exception as e:
            # error accessing the webcam, or processing the image
            raise e
    
    def _get_emotion(self, img) -> int:
        # Your code goes here
        #
        # img an np array of size NxN (square), each pixel is a value between 0 to 255
        # you have to resize this to image_size before sending to your model
        # to show the image here, you can use:
        # import matplotlib.pyplot as plt
        # plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        # plt.show()
        #
        # You have to use your saved model, use resized img as input, and get one classification value out of it
        # The classification value should be 0, 1, or 2 for neutral, happy or surprise respectively

        # return an integer (0, 1 or 2), otherwise the code will throw an error
        if not hasattr(self, '_emotion_model'):
            model_paths = [
                'results/hpo_best_trial_1_baseline_1771332002.keras',
                'model.keras',
            ]
            loaded_model = None
            for path in model_paths:
                try:
                    loaded_model = load_model(path)
                    break
                except Exception:
                    continue

            if loaded_model is None:
                raise FileNotFoundError('Cannot find a .keras model')

            self._emotion_model = loaded_model

        if img is None:
            raise ValueError('No image captured from webcam')

        frame = np.asarray(img)

        if not hasattr(self, '_face_detector'):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._face_detector = cv2.CascadeClassifier(cascade_path)

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        detected_faces = self._face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )

        if len(detected_faces) > 0:
            x, y, w, h = max(detected_faces, key=lambda face: face[2] * face[3])
            face_crop = gray[y:y + h, x:x + w]
        else:
            face_crop = gray

        tensor_img = tf.convert_to_tensor(face_crop, dtype=tf.float32)

        if len(tensor_img.shape) == 2:
            tensor_img = tf.expand_dims(tensor_img, axis=-1)
        elif len(tensor_img.shape) != 3:
            raise ValueError('Unexpected image shape: {}'.format(tensor_img.shape))

        resized = tf.image.resize(tensor_img, image_size)
        if resized.shape[-1] == 1:
            resized_rgb = tf.image.grayscale_to_rgb(resized)
        else:
            resized_rgb = resized[..., :3]

        batch = tf.expand_dims(resized_rgb, axis=0)

        prediction = self._emotion_model.predict(batch, verbose=0)
        predicted_idx = int(np.argmax(prediction, axis=-1)[0])

        if not hasattr(self, '_label_remap'):
            model_label_order = ['happy', 'neutral', 'surprise']
            self._label_remap = {
                model_idx: categories.index(label)
                for model_idx, label in enumerate(model_label_order)
            }

        emotion = int(self._label_remap.get(predicted_idx, predicted_idx))
        return emotion
    
    def get_move(self, board_state):
        row, col = None, None
        while row is None:
            row = self._get_row_or_col(True)
        while col is None:
            col = self._get_row_or_col(False)
        return row, col