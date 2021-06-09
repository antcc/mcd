"""
Try to "classify" samples based on random chance vs always guessing
the same category.
"""

import random


def random_vs_fixed(data, fix_mode="ApplyEyeMakeup", verbose=False):
    """Perform random and fixed classification.
       Extracted from `random_vs_fixed_mode.py`."""
    nb_random_matched = 0
    nb_mode_matched = 0

    # Try a random guess
    for item in data.data:
        choice = random.choice(data.classes)
        actual = item[1]

        if choice == actual:
            nb_random_matched += 1
        if actual == fix_mode:
            nb_mode_matched += 1

    random_accuracy = nb_random_matched/len(data.data)
    mode_accuracy = nb_mode_matched/len(data.data)

    if verbose:
        print("Randomly matched %.2f%%" % (random_accuracy*100))
        print("Mode matched %.2f%%" % (mode_accuracy*100))

    return random_accuracy*100, mode_accuracy*100
