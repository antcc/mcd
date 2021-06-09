"""
Train CNN
Based on:
https://keras.io/preprocessing/image/
and
https://keras.io/applications/
"""
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import (
    ModelCheckpoint, TensorBoard,
    EarlyStopping, CSVLogger)
import os.path
import time

timestamp = time.time()


def get_generators(classes):
    """Get image data generators."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join('data', 'train'),
        target_size=(299, 299),
        batch_size=8,
        classes=classes,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        os.path.join('data', 'test'),
        target_size=(299, 299),
        batch_size=8,
        classes=classes,
        class_mode='categorical')

    return train_generator, validation_generator


def get_model(nb_classes, weights='imagenet'):
    """Get pre-trained model."""
    # Create the base pre-trained model InceptionV3 with pre-trained imagenet
    # weights without including the top layers
    base_model = InceptionV3(weights=weights, include_top=False)
    x = base_model.output

    # Add a global spatial average pooling layer (Global average pooling
    # operation for spatial data, "GlobalAveragePooling2D layer")
    x = GlobalAveragePooling2D()(x)

    # Add a fully-connected layer (densely-connected NN layer, "Dense layer")
    # with 1024 units and relu activation
    x = Dense(1024, activation='relu')(x)

    # Add a logistic layer (densely-connected NN layer, "Dense layer") with the
    # number of video classes units and softmax activation
    predictions = Dense(nb_classes, activation='softmax')(x)

    # Define or compose the final model to train (groups layers into an object
    # with training and inference features, "The Model class")
    # with base_model.input as input and predictions as output
    model = Model(base_model.input, predictions, name="InceptionV3-finetune")

    return model


def freeze_all_but_top(model):
    """Train just the top 2 layers of the model."""
    # By default all layers are initialized as trainable
    # Select the non trainable layers, i.e. layer.trainable = False
    # In this case we freeze all convolutional InceptionV3 layers,
    # i.e. only the last two layers are trainable
    for layer in model.layers[:-2]:
        layer.trainable = False

    # Compile the model (should be done after setting layers to non-trainable)
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def freeze_all_but_mid_and_top(model):
    """After we fine-tune the dense layers, train deeper
       the mid and top layers."""
    # Select the trainable and non trainable layers, i.e.
    # layer.trainable = True or layer.trainable = False
    # In this case we choose to train the top inception blocks, i.e. we will
    # freeze the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False

    for layer in model.layers[172:]:
        layer.trainable = True

    # We need to recompile the model for these modifications to take effect
    # We use SGD with a low learning rate
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    model.fit(
        train_generator,
        steps_per_epoch=10,
        validation_data=validation_generator,
        validation_steps=10,
        epochs=nb_epoch,
        callbacks=callbacks)
    return model


def train(classes, weights_file=None):
    """Fine-tune the network."""
    nb_classes = len(classes)
    model = get_model(nb_classes)
    generators = get_generators(classes)

    if weights_file is None:
        print("Loading network from ImageNet weights.")
        # Get and train the top layers
        model = freeze_all_but_top(model)
        model = train_model(model, 10, generators)
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(
            'data',
            'checkpoints',
            ('UCF-' + str(nb_classes)
                + '-inception.{epoch:03d}-{val_loss:.2f}.hdf5')),
        verbose=1,
        save_best_only=True)

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=10)

    # Helper: TensorBoard
    tensorboard = TensorBoard(log_dir=os.path.join(
        'data', 'logs',
        'UCF-' + str(nb_classes) + '-inception'))

    # Helper: Save results.
    csv_logger = CSVLogger(
        os.path.join('data',
                     'logs',
                     ('UCF-' + str(nb_classes) + '-inception'
                         + '-training-' + str(timestamp) + '.log')))

    # Get and train the mid layers
    model = freeze_all_but_mid_and_top(model)
    model = train_model(model, 100, generators,
                        [checkpointer, early_stopper, tensorboard, csv_logger])

    return model
