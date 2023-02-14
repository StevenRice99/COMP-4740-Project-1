import argparse
import os

import torch

from main import main, get_name
from score import score


def perform(attempt: int, layers: int, width: int, epochs: int, batch: bool, dropout: bool, augment: bool):
    """
    Perform training for a given configuration.
    :param attempt: The attempt.
    :param layers: The number of hidden layers.
    :param width: The number of neurons in the hidden layers.
    :param epochs: The number of epochs to train for.
    :param batch: Apply batch normalization to hidden layers.
    :param dropout: Apply dropout to hidden layers.
    :param augment: Augment the training data.
    """
    path = f"{os.getcwd()}/Models/{get_name(batch, dropout, augment)}"
    current = f"{path} {attempt + 1}"
    # Nothing to do if the folder already exists.
    if os.path.exists(current):
        return
    # Train and rename the folder.
    main(layers, width, batch, dropout, augment, epochs, False)
    if os.path.exists(path):
        os.rename(path, f"{path} {attempt + 1}")


def looper(attempts: int, layers: int, width: int, epochs: int):
    """
    Loop for a given number of attempts for each possible network configuration.
    :param attempts: The number of attempts for each network configuration.
    :param layers: The number of hidden layers.
    :param width: The number of neurons in the hidden layers.
    :param epochs: The number of epochs to train for.
    """
    if attempts < 1:
        attempts = 1
    for attempt in range(attempts):
        print(f"Attempt {attempt + 1} of {attempts}.")
        perform(attempt, layers, width, epochs, False, False, False)
        perform(attempt, layers, width, epochs, False, False, True)
        perform(attempt, layers, width, epochs, False, True, False)
        perform(attempt, layers, width, epochs, False, True, True)
        perform(attempt, layers, width, epochs, True, False, False)
        perform(attempt, layers, width, epochs, True, False, True)
        perform(attempt, layers, width, epochs, True, True, False)
        perform(attempt, layers, width, epochs, True, True, True)
    score()


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="Wall-Following Robot Navigation")
        parser.add_argument("-a", "--attempts", type=int, help="The number of times to run.", default=10)
        parser.add_argument("-l", "--layers", type=int, help="The number of hidden layers.", default=6)
        parser.add_argument("-w", "--width", type=int, help="The number of neurons in the hidden layers.", default=128)
        parser.add_argument("-e", "--epoch", type=int, help="The number of epochs to train for.", default=100)
        a = vars(parser.parse_args())
        looper(a["attempts"], a["layers"], a["width"], a["epoch"])
    except KeyboardInterrupt:
        print("Training Stopped.")
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory. Try running with a smaller batch size.")
    except ValueError as error:
        print(error)
