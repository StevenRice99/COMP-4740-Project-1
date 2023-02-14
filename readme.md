# Wall-Following Robot Navigation

This is a solution for [Wall-Following Robot Navigation Data Set](https://archive.ics.uci.edu/ml/datasets/Wall-Following+Robot+Navigation+Data "Wall-Following Robot Navigation Data Set").

- [Setup](#setup "Setup")
- [Usage](#usage "Usage")
  - [Main](#main "Main")
  - [Score](#score "Score")
  - [Looper](#looper "Looper")
- [References](#references "References")

# Setup

1. Install [Python](https://www.python.org "Python").
   1. Python 3.10.7 was used for this project's creation, but any newer version should work provided it supports the required packages.
2. Clone or download and extract this repository.
3. Install required Python packages. If unfamiliar with Python, "pip" comes with standard Python installations, and you can run "pip *package*" to install a package.
   1. [PyTorch](https://pytorch.org "PyTorch")
      1. It is recommended you visit [PyTorch's get started page](https://pytorch.org/get-started/locally "PyTorch Get Started") which will allow you to select to install CUDA support if you have an Nvidia GPU. This will give you a command you can copy and run to install PyTorch, TorchVision, and TorchAudio, but feel free to remove TorchVision and TorchAudio from the command as they are not needed. Be sure to check which version of CUDA your Nvidia GPU supports.
      2. When running the script, a message will be output to the console if it is using CUDA.
   2. The remaining packages can be installed via "pip *package*" in the console:
      1. [numpy](https://numpy.org "numpy")
      2. [pandas](https://pandas.pydata.org "pandas")
      3. [sklearn](https://scikit-learn.org "scikit-learn")
      4. [tqdm](https://github.com/tqdm/tqdm "tqdm")

# Usage

## Main

1. Run "main.py" with the following optional parameters:
   1. -l, --layers - The number of hidden layers. Defaults to 6.
   2. -w, --width - The number of neurons in the hidden layers. Defaults to 128.
   3. -b, --batch - Apply batch normalization to hidden layers. Pass for true, otherwise false.
   4. -d, --dropout - Apply dropout to hidden layers. Pass for true, otherwise false.
   5. -a, --augment - Augment the training data. Pass for true, otherwise false.
   6. -e, --epoch - The number of epochs to train for. Defaults to 100.
   7. -t, --test - Test the model without doing any training. Pass for true, otherwise false.
2. Once done, a folder with the given model can be found in the "Models" folder which contains the following:
   1. "Model.pt" which contains the best weights and bias saved which can be loaded for inference as well as to continue training later.
   2. "Details.txt" which contains an overview of the model. 
   3. "Training.csv" which contains the loss and accuracy for each training epoch.

## Score

1. Run "score.py" which will generate "Results.csv" which contains the minimum, maximum, average, and standard deviations of the training.

## Looper

1. Run "looper.py" with the following optional parameters:
   1. -a, --attempts - The number of times to run. Defaults to 10.
   2. -l, --layers - The number of hidden layers. Defaults to 6.
   3. -w, --width - The number of neurons in the hidden layers. Defaults to 128.
   4. -e, --epoch - The number of epochs to train for. Defaults to 100.
2. Once done, folders as described in [main](#main "Main") will be generated the number of attempts each for every possible model combination of dropout, batch normalization, and data augmentation, as well as the scores of each calculated as described [score](#score "Score").

# References

- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.