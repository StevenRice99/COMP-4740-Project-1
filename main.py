import argparse
import os
import time

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class RobotDataset(Dataset):
    """
    The datasets built from the CSV data.
    """

    def __init__(self, inputs, labels, augment: bool):
        """
        Create the dataset.
        :param inputs: The inputs parsed from the CSV.
        :param labels: The labels parsed from the CSV.
        :param augment: If data should be augmented.
        """
        self.inputs = torch.tensor(inputs.to_numpy(), dtype=torch.float32)
        self.labels = torch.tensor(labels.to_numpy(), dtype=torch.int)
        self.augment = augment

    def __len__(self):
        """
        Get the length of the dataset.
        :return: The length of the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        Get an input and its label from the dataset.
        :param idx: The index to get.
        :return: The inputs and their label.
        """
        if not self.augment:
            return self.inputs[idx], self.labels[idx].type(torch.LongTensor)
        data = self.inputs[idx]
        data += (torch.rand(data.shape) - 0.5) * 2.0 * 0.01
        torch.clamp(data, 0.0, 1.0)
        return data, self.labels[idx].type(torch.LongTensor)


class NeuralNetwork(nn.Module):
    """
    The neural network to train on the dataset.
    """

    def __init__(self, hidden_number: int, hidden_size: int, batch_normalization: bool, dropout: bool):
        """
        Set up the neural network.
        :param hidden_number: The number of hidden layers.
        :param hidden_size: The size of hidden layers.
        :param batch_normalization: If batch normalization is enabled.
        :param dropout: If dropout is enabled.
        """
        super().__init__()
        # Define the input layer.
        self.layers = nn.Sequential(
            nn.Linear(24, 4 if hidden_number == 0 else hidden_size),
            nn.ReLU()
        )
        # Define the hidden layers.
        for i in range(hidden_number):
            # Optionally apply batch normalization.
            if batch_normalization:
                self.layers.append(nn.BatchNorm1d(hidden_size))
            # Main part of the hidden layers.
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
            # Optionally apply dropout.
            if dropout:
                self.layers.append(nn.Dropout())
        # Define the output layer.
        self.layers.append(nn.Linear(24 if hidden_number == 0 else hidden_size, 4))
        self.layers.append(nn.ReLU())
        # Define the loss.
        self.loss = nn.CrossEntropyLoss()
        # Define the optimizer.
        self.optimizer = optim.Adam(self.parameters())
        # Run on GPU if available.
        self.to(get_processing_device())

    def forward(self, values):
        """
        Feed forward an image into the neural network.
        :param values: The values as a proper tensor.
        :return: The final output layer from the network.
        """
        return self.layers(values)

    def predict(self, values):
        """
        Get the network's prediction for an image.
        :param values: The values as a proper tensor.
        :return: The number the network predicts for this image.
        """
        with torch.no_grad():
            # Get the highest confidence output value.
            return torch.argmax(self.forward(values), axis=-1)

    def optimize(self, values, label):
        """
        Optimize the neural network to fit the training data.
        :param values: The values as a proper tensor.
        :param label: The label of the image.
        :return: The network's loss on this prediction.
        """
        self.optimizer.zero_grad()
        loss = self.loss(self.forward(values), label)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def dataset_details(title: str, labels):
    """
    Output dataset details to the console.
    :param title: The title of the dataset to tell if this is the training or testing dataset.
    :param labels: The labels.
    :return: The total count of the dataset.
    """
    counts = [0, 0, 0, 0, 0, 0, 0]
    for label in labels:
        counts[label] += 1
    total = sum(counts)
    print(f"{title} Dataset: {total}\n"
          f"Move-Forward:      {counts[0]:>5}\t{counts[0] / total * 100}%\n"
          f"Slight-Right-Turn: {counts[1]:>5}\t{counts[1] / total * 100}%\n"
          f"Sharp-Right-Turn:  {counts[2]:>5}\t{counts[2] / total * 100}%\n"
          f"Slight-Left-Turn:  {counts[3]:>5}\t{counts[3] / total * 100}%")
    return total


def get_processing_device():
    """
    Get the device to use for training, so we can use the GPU if CUDA is available.
    :return: The device to use for training being a CUDA GPU if available, otherwise the CPU.
    """
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_tensor(tensor, device=get_processing_device()):
    """
    Convert an image to a tensor to run on the given device.
    :param tensor: The data to convert to a tensor.
    :param device: The device to use for training being a CUDA GPU if available, otherwise the CPU.
    :return: The data ready to be used.
    """
    return tensor.to(device)


def test(model, batch: int, dataloader):
    """
    Test a neural network.
    :param model: The neural network.
    :param batch: The batch size.
    :param dataloader: The dataloader to test.
    :return: The model's accuracy.
    """
    # Switch to evaluation mode.
    model.eval()
    # Count how many are correct.
    correct = 0
    # Loop through all data.
    for raw_image, raw_label in dataloader:
        image, label = to_tensor(raw_image), to_tensor(raw_label)
        # If properly predicted, count it as correct.
        correct += (label == model.predict(image)).sum().item()
    # Calculate the overall accuracy.
    return correct / (len(dataloader) * batch) * 100.0


def save(name: str, model, best_model, epoch: int, no_change: int, best_accuracy: float, loss: float):
    """
    Save the model.
    :param name: The name of the model.
    :param model: The model.
    :param best_model: The best model.
    :param epoch: The epoch.
    :param no_change: The number of epochs without improvement.
    :param best_accuracy: The best accuracy of the best model.
    :param loss: The loss of the model.
    """
    torch.save({
        'Best': best_model,
        'Training': model.state_dict(),
        'Optimizer': model.optimizer.state_dict(),
        'Epoch': epoch,
        'No Change': no_change,
        'Best Accuracy': best_accuracy,
        'Loss': loss
    }, f"{os.getcwd()}/Models/{name}/Model.pt")


def write_parameters(name: str, best_accuracy: float, train_accuracy: float, inference_time: float, trainable_parameters: int, best_epoch: int):
    """
    Write some useful parameters to a text file.
    :param name: The name of the model.
    :param best_accuracy: The best accuracy the model achieved.
    :param train_accuracy: The training accuracy.
    :param inference_time: The average inference time.
    :param trainable_parameters: The number of trainable parameters.
    :param best_epoch: The epoch which the best accuracy was achieved on.
    """
    f = open(f"{os.getcwd()}/Models/{name}/Details.txt", "w")
    f.write(f"Testing Accuracy: {best_accuracy}\n"
            f"Training Accuracy: {train_accuracy}\n"
            f"Average Inference Time: {inference_time} ms\n"
            f"Trainable Parameters: {trainable_parameters}\n"
            f"Best Epoch: {best_epoch}")
    f.close()


def get_name(batch: bool, dropout: bool, augment: bool):
    """
    Get the name of the current model.
    :param batch: If batch normalization is being used.
    :param dropout: If dropout is being used.
    :param augment: If augmentation is being used.
    :return: The name of the model.
    """
    name = "Network"
    if augment:
        name += " Augmented"
    if batch:
        name += " Batch Normalization"
    if dropout:
        name += " Dropout"
    return name


def main(layers: int, width: int, batch: bool, dropout: bool, augment: bool, epochs: int, load: bool):
    """
    Main program execution.
    :param layers: The number of hidden layers.
    :param width: The number of neurons in the hidden layers.
    :param batch: Apply batch normalization to hidden layers.
    :param dropout: Apply dropout to hidden layers.
    :param augment: Augment the training data.
    :param epochs: The number of epochs to train for.
    :param load: Whether to load an existing model or train a new one.
    :return: Nothing.
    """
    # Set the name.
    name = get_name(batch, dropout, augment)
    print(f"Wall-Following Robot Navigation")
    print(f"Running on GPU with CUDA {torch.version.cuda}." if torch.cuda.is_available() else "Running on CPU.")
    data_path = f"{os.getcwd()}/sensor_readings_24.data"
    if not os.path.exists(data_path):
        print("Dataset missing.")
        return
    # Setup datasets.
    print("Loading data...")
    df = pd.read_csv(data_path, header=None)
    # Convert the label into a numerical value.
    df[24] = df[24].map({"Move-Forward": 0, "Slight-Right-Turn": 1, "Sharp-Right-Turn": 2, "Slight-Left-Turn": 3})
    # Min-max normalize the data between zero and one.
    max_value = -100000
    min_value = 100000
    for i in range(24):
        for j in range(len(df[0])):
            if df.at[j, i] > max_value:
                max_value = df.at[j, i]
            elif df.at[j, i] < min_value:
                min_value = df.at[j, i]
    difference = max_value - min_value
    for i in range(24):
        for j in range(len(df[0])):
            df.at[j, i] = (df.at[j, i] - min_value) / difference
    # Split the data consistently every time.
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:24], df.iloc[:, 24], test_size=0.20, random_state=1)
    # Create the datasets.
    training_data = RobotDataset(x_train, y_train, augment)
    testing_data = RobotDataset(x_test, y_test, False)
    # Log details of the datasets.
    training_total = dataset_details("Training", y_train)
    testing_total = dataset_details("Testing", y_test)
    # Create the data loaders.
    training = DataLoader(training_data, batch_size=64, shuffle=True)
    testing = DataLoader(testing_data, batch_size=64, shuffle=False)
    print(name)
    # Create the model.
    model = NeuralNetwork(layers, width, batch, dropout)
    # Load a model if flagged to do so.
    if load:
        # If a model does not exist to load decide to generate a new model instead.
        if not os.path.exists(f"{os.getcwd()}/Models/{name}/Model.pt"):
            print(f"Model '{name}' does not exist to load.")
            return
        try:
            saved = torch.load(f"{os.getcwd()}/Models/{name}/Model.pt")
            model.load_state_dict(saved['Best'])
        except:
            print("Model to load has different structure than 'model_builder'.py, cannot load.")
            return
        train_accuracy = test(model, 64, training)
        start = time.time_ns()
        accuracy = test(model, 64, testing)
        end = time.time_ns()
        inference_time = ((end - start) / testing_total) / 1e+6
        print(f"Testing Accuracy = {accuracy}%\n"
              f"Training Accuracy = {train_accuracy}%\n"
              f"Average Inference Time: {inference_time} ms")
        return
    # Otherwise, train a model.
    best_model = model.state_dict()
    # Check if an existing model of the same name exists.
    if os.path.exists(f"{os.getcwd()}/Models/{name}/Model.pt"):
        try:
            print(f"Model '{name}' already exists, attempting to load to continue training...")
            saved = torch.load(f"{os.getcwd()}/Models/{name}/Model.pt")
            best_model = saved['Best']
            model.load_state_dict(saved['Training'])
            model.optimizer.load_state_dict(saved['Optimizer'])
            epoch = saved['Epoch']
            no_change = saved['No Change']
            best_accuracy = saved['Best Accuracy']
            loss = saved['Loss']
            print(f"Continuing training for '{name}' from epoch {epoch} for {epochs} epochs.")
        except:
            print("Unable to load training data, exiting.")
            return
    else:
        loss = -1
        epoch = 1
        no_change = 0
        print(f"Starting training for {epochs} epochs.")
    # Ensure folder to save models exists.
    if not os.path.exists(f"{os.getcwd()}/Models"):
        os.mkdir(f"{os.getcwd()}/Models")
    if not os.path.exists(f"{os.getcwd()}/Models/{name}"):
        os.mkdir(f"{os.getcwd()}/Models/{name}")
    start = time.time_ns()
    accuracy = test(model, 64, testing)
    end = time.time_ns()
    inference_time = ((end - start) / testing_total) / 1e+6
    # If new training, write initial files.
    if epoch == 1:
        best_accuracy = accuracy
        f = open(f"{os.getcwd()}/Models/{name}/Training.csv", "w")
        f.write("Epoch,Loss,Training,Testing")
        f.close()
    train_accuracy = test(model, 64, training)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Network parameters: {trainable_parameters}")
    write_parameters(name, best_accuracy, train_accuracy, inference_time, trainable_parameters, 0)
    save(name, model, best_model, epoch, no_change, best_accuracy, loss)
    # Train for set epochs.
    while True:
        if epoch > epochs:
            print("Training finished.")
            return
        loss_message = "Loss = " + (f"{loss:.4}" if epoch > 1 else "N/A")
        improvement = "Improvement " if no_change == 0 else f"{no_change} Epochs No Improvement "
        msg = f"Epoch {epoch}/{epochs} | {loss_message} | Train = {train_accuracy:.4}% | Test = {accuracy:.4}% | Best = {best_accuracy:.4}% | {improvement}"
        # Reset loss every epoch.
        loss = 0
        # Switch to training.
        model.train()
        for raw_image, raw_label in tqdm(training, msg):
            image, label = to_tensor(raw_image), to_tensor(raw_label)
            loss += model.optimize(image, label)
        loss /= training_total
        # Check how well the newest epoch performs.
        start = time.time_ns()
        accuracy = test(model, 64, testing)
        end = time.time_ns()
        train_accuracy = test(model, 64, training)
        # Check if this is the new best model.
        if accuracy > best_accuracy:
            best_model = model.state_dict()
            best_accuracy = accuracy
            inference_time = ((end - start) / testing_total) / 1e+6
            no_change = 0
            write_parameters(name, best_accuracy, train_accuracy, inference_time, trainable_parameters, epoch)
        else:
            no_change += 1
        # Save data.
        f = open(f"{os.getcwd()}/Models/{name}/Training.csv", "a")
        f.write(f"\n{epoch},{loss},{train_accuracy},{accuracy}")
        f.close()
        epoch += 1
        save(name, model, best_model, epoch, no_change, best_accuracy, loss)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="Wall-Following Robot Navigation")
        parser.add_argument("-l", "--layers", type=int, help="The number of hidden layers.", default=6)
        parser.add_argument("-w", "--width", type=int, help="The number of neurons in the hidden layers.", default=128)
        parser.add_argument("-b", "--batch", help="Apply batch normalization to hidden layers.", action="store_true")
        parser.add_argument("-d", "--dropout", help="Apply dropout to hidden layers.", action="store_true")
        parser.add_argument("-a", "--augment", help="Augment the training data.", action="store_true")
        parser.add_argument("-e", "--epoch", type=int, help="The number of epochs to train for.", default=100)
        parser.add_argument("-t", "--test", help="Test the model without doing any training.", action="store_true")
        a = vars(parser.parse_args())
        main(a["layers"], a["width"], a["batch"], a["dropout"],a["augment"], a["epoch"], a["test"])
    except KeyboardInterrupt:
        print("Training Stopped.")
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory. Try running with a smaller batch size.")
    except ValueError as error:
        print(error)
