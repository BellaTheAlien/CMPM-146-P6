import csv
import itertools
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.optimizers import Adam

from config import categories, image_size
from preprocess import get_datasets

input_shape = (image_size[0], image_size[1], 3)
categories_count = len(categories)

def plot_history(history):
    history_dict = history.history if hasattr(history, 'history') else history

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(24, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()

def build_model(model_config):
    num_conv_layers = model_config['num_conv_layers']
    num_fc_layers = model_config['num_fc_layers']
    dropout_count = model_config['dropout_count']
    dropout_location = model_config['dropout_location']
    dropout_rate = model_config['dropout_rate']
    learning_rate = model_config['learning_rate']

    model_layers = [Rescaling(1.0 / 255, input_shape=input_shape)]

    for conv_index in range(num_conv_layers):
        filters = min(16 * (2 ** conv_index), 128)
        model_layers.append(layers.Conv2D(filters, (3, 3), activation='relu'))
        model_layers.append(layers.MaxPooling2D(2, 2))

    if dropout_count >= 1 and dropout_location in ['conv', 'conv+dense']:
        model_layers.append(layers.Dropout(dropout_rate))

    model_layers.append(layers.Flatten())

    dense_dropout_inserted = 0
    for fc_index in range(num_fc_layers):
        units = 512 if fc_index == 0 else 256
        model_layers.append(layers.Dense(units, activation='relu'))

        should_add_dense_dropout = (
            dropout_count >= 1 and
            dropout_location in ['dense', 'conv+dense', 'dense+dense'] and
            dense_dropout_inserted < (2 if dropout_location == 'dense+dense' else 1)
        )

        if should_add_dense_dropout:
            model_layers.append(layers.Dropout(dropout_rate))
            dense_dropout_inserted += 1

    model_layers.append(layers.Dense(categories_count, activation='softmax'))

    model = Sequential(model_layers)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Hyper-parameter optimization 
def generate_trial_configs(max_trials=24, seed=47):
    search_space = {
        'num_conv_layers': [2, 3, 4],
        'num_fc_layers': [1, 2],
        'dropout_count': [0, 1, 2],
        'dropout_location': {
            0: ['none'],
            1: ['conv', 'dense'],
            2: ['conv+dense', 'dense+dense']
        },
        'dropout_rate': [0.2, 0.35, 0.5],
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'epochs': [20, 30, 40],
    }

    all_configs = []
    for num_conv_layers, num_fc_layers, dropout_count, learning_rate, epochs in itertools.product(
        search_space['num_conv_layers'],
        search_space['num_fc_layers'],
        search_space['dropout_count'],
        search_space['learning_rate'],
        search_space['epochs'],
    ):
        valid_locations = search_space['dropout_location'][dropout_count]
        rates = [0.0] if dropout_count == 0 else search_space['dropout_rate']

        for dropout_location, dropout_rate in itertools.product(valid_locations, rates):
            all_configs.append({
                'num_conv_layers': num_conv_layers,
                'num_fc_layers': num_fc_layers,
                'dropout_count': dropout_count,
                'dropout_location': dropout_location,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'epochs': epochs,
            })

    random.Random(seed).shuffle(all_configs)
    return all_configs[:min(max_trials, len(all_configs))]

def write_trials_csv(trials, csv_path):
    fieldnames = [
        'trial_id', 'val_accuracy', 'best_epoch', 'test_accuracy',
        'num_conv_layers', 'num_fc_layers', 'dropout_count',
        'dropout_location', 'dropout_rate', 'learning_rate', 'epochs'
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)


if __name__ == "__main__":
    tf.random.set_seed(47)
    np.random.seed(47)
    random.seed(47)

    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_datasets()

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=4,
        restore_best_weights=True
    )

    trial_configs = generate_trial_configs(max_trials=24, seed=47)
    print(f'* Running hyper-parameter search with {len(trial_configs)} trials')

    best_trial = None
    best_model = None
    best_history = None
    all_trials = []

    for trial_id, config in enumerate(trial_configs, start=1):
        print(f"\n* Trial {trial_id}/{len(trial_configs)}: {config}")
        model = build_model(config)

        history = model.fit(
            x=train_dataset,
            validation_data=validation_dataset,
            epochs=config['epochs'],
            callbacks=[early_stopping],
            verbose='auto'
        )

        val_history = history.history['val_accuracy']
        best_epoch_index = int(np.argmax(val_history))
        best_epoch = best_epoch_index + 1
        best_val_accuracy = float(val_history[best_epoch_index])

        _, test_accuracy = model.evaluate(test_dataset, verbose=0)

        trial_result = {
            'trial_id': trial_id,
            'val_accuracy': best_val_accuracy,
            'best_epoch': best_epoch,
            'test_accuracy': float(test_accuracy),
            **config,
        }
        all_trials.append(trial_result)

        print(
            f"  -> best val_acc={best_val_accuracy:.4f} at epoch {best_epoch}, "
            f"test_acc={test_accuracy:.4f}"
        )

        if best_trial is None or best_val_accuracy > best_trial['val_accuracy']:
            best_trial = trial_result
            best_model = model
            best_history = history.history

    timestamp = int(time.time())
    os.makedirs('results', exist_ok=True)
    run_prefix = f"hpo_best_trial_1_baseline_{timestamp}"

    best_model_path = f"results/{run_prefix}.keras"
    best_history_path = f"results/{run_prefix}_history.npy"
    trials_csv_path = f"results/{run_prefix}_trials.csv"
    summary_path = f"results/{run_prefix}_summary.txt"

    best_model.save(best_model_path)
    np.save(best_history_path, best_history)

    sorted_trials = sorted(all_trials, key=lambda item: item['val_accuracy'], reverse=True)
    write_trials_csv(sorted_trials, trials_csv_path)

    predictions = best_model.predict(test_dataset, verbose=0)
    labels = np.concatenate([y for _, y in test_dataset], axis=0)
    y_pred = np.argmax(predictions, axis=-1)
    y_true = np.argmax(labels, axis=-1)
    cm = confusion_matrix(y_true, y_pred)

    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write('Hyper-parameter Optimization Summary\n')
        summary_file.write('===================================\n')
        summary_file.write(f"Best trial id: {best_trial['trial_id']}\n")
        summary_file.write(f"Best validation accuracy: {best_trial['val_accuracy']:.4f}\n")
        summary_file.write(f"Best epoch (overfitting point proxy): {best_trial['best_epoch']}\n")
        summary_file.write(f"Test accuracy (best-val model): {best_trial['test_accuracy']:.4f}\n")
        summary_file.write('\nBest hyper-parameters:\n')
        summary_file.write(f"  num_conv_layers: {best_trial['num_conv_layers']}\n")
        summary_file.write(f"  num_fc_layers: {best_trial['num_fc_layers']}\n")
        summary_file.write(f"  dropout_count: {best_trial['dropout_count']}\n")
        summary_file.write(f"  dropout_location: {best_trial['dropout_location']}\n")
        summary_file.write(f"  dropout_rate: {best_trial['dropout_rate']}\n")
        summary_file.write(f"  learning_rate: {best_trial['learning_rate']}\n")
        summary_file.write(f"  max_epochs: {best_trial['epochs']}\n")
        summary_file.write('\nConfusion Matrix:\n')
        summary_file.write(str(cm))
        summary_file.write('\n')

    print('\n* Hyper-parameter search complete')
    print(f"* Best validation accuracy: {best_trial['val_accuracy']:.4f}")
    print(f"* Test accuracy with best-val model: {best_trial['test_accuracy']:.4f}")
    print('* Best trial hyper-parameters:')
    print({
        'num_conv_layers': best_trial['num_conv_layers'],
        'num_fc_layers': best_trial['num_fc_layers'],
        'dropout_count': best_trial['dropout_count'],
        'dropout_location': best_trial['dropout_location'],
        'dropout_rate': best_trial['dropout_rate'],
        'learning_rate': best_trial['learning_rate'],
        'epochs': best_trial['epochs'],
        'best_epoch': best_trial['best_epoch'],
    })
    print(f"* Saved best model to {best_model_path}")
    print(f"* Saved best history to {best_history_path}")
    print(f"* Saved all trials to {trials_csv_path}")
    print(f"* Saved summary to {summary_path}")

    if best_trial['test_accuracy'] >= 0.70:
        print('* Target reached: test accuracy is at least 70%.')
    else:
        print('* Target not yet reached: test accuracy is below 70%.')

    plot_history(best_history)
