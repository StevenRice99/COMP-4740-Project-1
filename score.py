import os
import statistics

import pandas as pd

from main import get_name


def calculate(augment: bool, batch: bool, dropout: bool, s: str):
    """
    Calculate the scores of a network configuration.
    :param augment: Augment the training data.
    :param batch: Apply batch normalization to hidden layers.
    :param dropout: Apply dropout to hidden layers.
    :param s: If this is checking for accuracy or precision.
    """
    results = {"Training": [], "Testing": []}
    root = f"{os.getcwd()}/Models"
    name = get_name(batch, dropout, augment)
    # Check all results.
    for f in os.listdir(root):
        path = os.path.join(root, f)
        # Only paths for this model.
        model = os.path.basename(path)
        model = ''.join([c for c in model if not c.isdigit()])
        model = model.strip()
        if name != model:
            continue
        # If there is no CSV, nothing to read.
        path = os.path.join(path, "Training.csv")
        if not os.path.exists(path):
            continue
        # Read the results.
        df = pd.read_csv(path)
        results["Training"].append(df[f"Training {s}"].max())
        results["Testing"].append(df[f"Testing {s}"].max())
    # Update the name to be shorter.
    name = name.replace("Network Augmented", "A")
    name = name.replace("Network", "N")
    name = name.replace("Batch Normalization", "B")
    name = name.replace("Dropout", "D")
    # Calculate values.
    return None if len(results["Training"]) == 0 else {
        "Name": name,
        "Min": {"Training": min(results["Training"]), "Testing": min(results["Testing"])},
        "Max": {"Training": max(results["Training"]), "Testing": max(results["Testing"])},
        "Avg": {"Training": statistics.mean(results["Training"]), "Testing": statistics.mean(results["Testing"])},
        "Std": {"Training": statistics.stdev(results["Training"]) if len(results["Training"]) > 1 else "-",
                "Testing": statistics.stdev(results["Testing"]) if len(results["Testing"]) > 1 else "-"}}


def score():
    """
    Calculate the scores of all network configuration.
    """
    # Get all results.
    for i in range(2):
        s = "Accuracy" if i == 0 else "Precision"
        results = []
        result = calculate(False, False, False, s)
        if result is not None:
            results.append(result)
        result = calculate(False, False, True, s)
        if result is not None:
            results.append(result)
        result = calculate(False, True, False, s)
        if result is not None:
            results.append(result)
        result = calculate(False, True, True, s)
        if result is not None:
            results.append(result)
        result = calculate(True, False, False, s)
        if result is not None:
            results.append(result)
        result = calculate(True, False, True, s)
        if result is not None:
            results.append(result)
        result = calculate(True, True, False, s)
        if result is not None:
            results.append(result)
        result = calculate(True, True, True, s)
        if result is not None:
            results.append(result)
        # Nothing to do if no results.
        if len(results) == 0:
            return
        # Write data to the file.
        f = open(f"{os.getcwd()}/{s}.csv", "w")
        f.write("Model")
        for result in results:
            f.write(f",{result['Name']},{result['Name']}")
        f.write("\nMetric")
        for _ in results:
            f.write(",Training,Testing")
        f.write("\n")
        f.write("Min")
        for result in results:
            f.write(f",{result['Min']['Training']},{result['Min']['Testing']}")
        f.write("\n")
        f.write("Max")
        for result in results:
            f.write(f",{result['Max']['Training']},{result['Max']['Testing']}")
        f.write("\n")
        f.write("Average")
        for result in results:
            f.write(f",{result['Avg']['Training']},{result['Avg']['Testing']}")
        f.write("\n")
        f.write("STD")
        for result in results:
            f.write(f",{result['Std']['Training']},{result['Std']['Testing']}")
        f.close()


if __name__ == '__main__':
    score()
