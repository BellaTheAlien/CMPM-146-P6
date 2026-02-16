from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        self.model = Sequential([ Rescaling(1./255, input_shape=input_shape),
                                  layers.Rescaling(1./255),

                                  # Add convolutional and max pooling layers; 150x150 into 75x75
                                  layers.Conv2D(16, (3, 3), activation='relu'),
                                  layers.MaxPooling2D(2, 2),
                                  # 75x75 into 37x37
                                  layers.Conv2D(32, (3, 3), activation='relu'),
                                  layers.MaxPooling2D(2, 2),

                                  # 37x37 into 18x18
                                  layers.Conv2D(64, (3, 3), activation='relu'),
                                  layers.MaxPooling2D(2, 2),

                                  # flatten and add a fully connected layer with softmax
                                  # 512 neurons to learn more complex features
                                  layers.Flatten(),
                                  layers.Dense(512, activation='relu'),
                                  layers.Dense(categories_count, activation='softmax')])
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )