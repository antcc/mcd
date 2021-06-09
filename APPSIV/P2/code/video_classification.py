# -*- coding: utf-8 -*-
"""
LAB 2: Video Classification.

Antonio Coín Castro
May 16th 2021

NOTE: This script should be run from the root directory of
the project, without modifying the provided file structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import sys
import os

"""
The raw videos extracted from the *UCF-20* dataset should be present in
directory `data/`. Then, to complete data preparation, we run the following
scripts:

```
!python 1_move_files.py
!python 2_extract_files.py
```

In the following experiments we will consider 5 frames per video and a variable
number of classes. To activate CNN training, set `TRAIN_CNN = True`. If the
models are re-trained, the corresponding logs and checkpoints need to be
specified.
"""

##
# Initial preparations and global variables
##

# Whether to train the models
TRAIN_CNN = False  # <-- Change to re-train

# Silence tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Append code paths
CODE = "code/"
sys.path.append(CODE)
sys.path.append(CODE + "S1/")
sys.path.append(CODE + "S2/")

# Set random seed
random.seed(2021)
np.random.seed(2021)

##
# Utility functions
##

from data import DataSet


def get_data(class_limit=5, seq_length=5):
    """Get video data from a few classes.
        - seq_length = (int) the number of frames to consider for each video
        - class_limit = (int in [1, 101] or None) number of classes
              to limit the data to [None = no limit].
    """
    return DataSet(seq_length, class_limit)


##
# Session 1: Random vs fixed mode
##

from random_vs_fixed_mode import random_vs_fixed

"""
We will try to "classify" samples using random and fixed assignments.
For this task we will only use the first 5 classes of UCF.
"""

print("--> SESSION 1: RANDOM VS FIXED MODE <--")

# Get data for analysis
data = get_data()
data_info = np.array(data.data)
classes = data.classes
N = np.sum(np.array(data.data)[:, -1].astype(np.int16))
C = len(classes)
print(f"\n{N} samples in {C} different classes:")
for c in classes:
    print(f"  {c}")

### Random strategy

M = 1000
random_accs = []

for _ in range(M):
    random_acc, _ = random_vs_fixed(data)
    random_accs.append(random_acc)

print(f"\nRandom strategy in {M} independent runs...")
print(f"  Min random accuracy: {np.min(random_accs):.2f}%")
print(f"  Max random accuracy: {np.max(random_accs):.2f}%")
print(f"  Mean random accuracy: {np.mean(random_accs):.2f}%")
print(f"  Median random accuracy: {np.median(random_accs):.2f}%")

### Fixed strategy

mode_accs = []

print("\nFixed strategy...")
for c in classes:
    _, mode_acc = random_vs_fixed(data, fix_mode=c)
    mode_accs.append(mode_acc)
    print(f"  {c}: {mode_acc:.2f}%")

### Results analysis

print("\nResults analysis...")

# Get class distribution
class_distr = dict.fromkeys(data.classes, 0)
for sample_group in data_info:
    class_name = sample_group[1]
    nb_samples = sample_group[-1].astype(np.int16)
    class_distr[class_name] += 100*nb_samples/N

plt.figure(figsize=(7, 4))
plt.title("Class distribution")
plt.ylabel("% of samples")
plt.bar(class_distr.keys(), class_distr.values(), width=0.5,
        color=cm.tab10.colors[:C])
plt.show()

plt.figure(figsize=(5, 4))
plt.title("Accuracy of fixed and random choices")
plt.xlabel("Accuracy (%)")
plt.barh(np.concatenate((classes, ["Random"])),
         mode_accs + [np.mean(random_accs)],
         color=cm.tab10.colors[:C + 1],
         height=0.5)
plt.show()


##
# Session 2: Frame-by-frame CNN classification
##

from plot_train_cnnlog import plot_acc_log, plot_loss_log
from validate_cnn import spot_check
from train_cnn import train

"""
We will develop a frame-by-frame CNN classifier using a pretrained network and
*fine-tuning* techniques. Then, we will compare the performance of this
classifier for the first 5, 10, 15 and 20 classes of UCF. Moreover, we will
also compare the results obtained with the random and fixed modes in terms of
accuracy.
"""

print("\n--> SESSION 2: FRAME-BY-FRAME CNN CLASSIFICATION <--")

### 5 classes

# Get data
print("\n=> UCF-5")
data = get_data(class_limit=5)
N = np.sum(np.array(data.data)[:, -1].astype(np.int16))
C = len(data.classes)
print(f"{N} samples in {C} different classes:")
for c in data.classes:
    print(f"  {c}")

# Train model
if TRAIN_CNN:
    model = train(data.classes)
    # Check model structure
    print(model.summary())

# Validate CNN
print("\nUCF-5 CNN predictions")
spot_check(data.classes, "data/checkpoints/UCF-5-inception.009-0.14.hdf5")

# View training logs
print("\nResults analysis...")
plot_acc_log("data/logs/UCF-5-inception-training-1621098839.8652656.log", 5)
plot_loss_log("data/logs/UCF-5-inception-training-1621098839.8652656.log", 5)

"""
As we can see, the accuracy obtained with the CNN is far greater than the ones
obtained with either random or fixed mode classification.
"""

### 10 classes

# Get data
print("\n=> UCF-10")
data = get_data(class_limit=10)
N = np.sum(np.array(data.data)[:, -1].astype(np.int16))
C = len(data.classes)
print(f"{N} samples in {C} different classes:")
for c in data.classes:
    print(f"  {c}")

# Train model
if TRAIN_CNN:
    _ = train(data.classes)

# Validate CNN
print("\nUCF-10 CNN predictions")
spot_check(data.classes, "data/checkpoints/UCF-10-inception.005-0.20.hdf5")

# View training logs
print("\nResults analysis...")
plot_acc_log("data/logs/UCF-10-inception-training-1621099869.2524738.log", 10)
plot_loss_log("data/logs/UCF-10-inception-training-1621099869.2524738.log", 10)

### 15 classes

# Get data
print("\n=> UCF-15")
data = get_data(class_limit=15)
N = np.sum(np.array(data.data)[:, -1].astype(np.int16))
C = len(data.classes)
print(f"{N} samples in {C} different classes:")
for c in data.classes:
    print(f"  {c}")

# Train model
if TRAIN_CNN:
    _ = train(data.classes)

# Validate CNN
print("\nUCF-15 CNN predictions")
spot_check(data.classes, "data/checkpoints/UCF-15-inception.014-0.25.hdf5")

# View training logs
print("\nResults analysis...")
plot_acc_log("data/logs/UCF-15-inception-training-1621099869.2524738.log", 15)
plot_loss_log("data/logs/UCF-15-inception-training-1621099869.2524738.log", 15)

### 20 classes

# Get data
print("\n=> UCF-20")
data = get_data(class_limit=20)
N = np.sum(np.array(data.data)[:, -1].astype(np.int16))
C = len(data.classes)
print(f"{N} samples in {C} different classes:")
for c in data.classes:
    print(f"  {c}")

# Train model
if TRAIN_CNN:
    _ = train(data.classes)

# Validate CNN
print("\nUCF-20 CNN predictions")
spot_check(data.classes, "data/checkpoints/UCF-20-inception.009-0.56.hdf5")

# View training logs
print("\nResults analysis...")
plot_acc_log("data/logs/UCF-20-inception-training-1621099869.2524738.log", 20)
plot_loss_log("data/logs/UCF-20-inception-training-1621099869.2524738.log", 20)
