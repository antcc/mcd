"""
Classify some images
"""
import numpy as np
import operator
import random
import matplotlib.pyplot as plt
import glob
import os.path
from processor import process_image
from keras.models import load_model


def spot_check(classes, checkpoint, nb_images=5, plot_img=False):
    """Classify `nb_images` random images from a model
       represented by a checkpoint file."""

    model = load_model(checkpoint)

    # Get all our test images.
    images = []
    dir_list = ["data/test/" + c for c in classes]
    for test_dir in dir_list:
        images.extend(glob.glob(os.path.join(test_dir, '*.jpg')))

    for _ in range(nb_images):
        print('-'*80)
        # Get a random row.
        sample = random.randint(0, len(images) - 1)
        image = images[sample]

        # Turn the image into an array.
        print(image)
        image_arr = process_image(image, (299, 299, 3))
        if plot_img:
            plt.imshow(image_arr)
            plt.show()
        image_arr = np.expand_dims(image_arr, axis=0)

        # Predict.
        predictions = model.predict(image_arr)

        # Show how much we think it's each one.
        label_predictions = {}
        for i, label in enumerate(classes):
            label_predictions[label] = predictions[0][i]

        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True)

        for i, class_prediction in enumerate(sorted_lps):
            # Just get the top five.
            if i > 4:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            i += 1
