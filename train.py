import os
import datetime
import numpy as np
from model import create_resnet101_segmentation_model
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from utils import load_images_and_masks, transform_dataset, display_transformation

# Input shape must be a factor of 2 and W==H.
# If want to try a shape that is not factor of 2 or W!=H, change the model accordingly
# in the upsampling section
INPUT_SHAPE = (256, 256, 3)
BATCH_SIZE = 8
MAX_EPOCHS = 5
NUM_CLASSES = 6

# Set up data paths
working_dir = os.getcwd()

data_dirs = {
    "train": os.path.join(working_dir, "final_data", "train"),
    "valid": os.path.join(working_dir, "final_data", "valid"),
    "test": os.path.join(working_dir, "final_data", "test")
}

# Figure directory to save figures
figure_dir = os.path.join(working_dir, "figures")
os.makedirs(figure_dir, exist_ok=True)

# Figure directory to save figures
chkpt_dir = os.path.join(working_dir, "checkpoints")
os.makedirs(figure_dir, exist_ok=True)

def train():

    """
    A function to perform training by initializing the values of the model object.
    ModelCheckpoint is used to save the trained weights.
    fit_generator starts the training process.

    """
    # Load data
    data_dict = {}

    for key, dir_path in data_dirs.items():
        images, masks = load_images_and_masks(os.path.join(dir_path, "images"), os.path.join(dir_path, "masks"))  # Assuming image and mask directories have the same structure
        masks = np.expand_dims(masks, axis=-1)

        # Apply augmentation
        images, masks = transform_dataset(images, masks, input_shape=(INPUT_SHAPE[0], INPUT_SHAPE[1]))

        data_dict[key + "_images"] = images
        data_dict[key + "_masks"] = masks


    # Initialize model
    model = create_resnet101_segmentation_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, layer_no=0, weights='imagenet', include_top=False)
    uid = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"Unique Identifier: {uid}")
    plot_model(model, to_file=os.path.join(figure_dir, f"ResNet101_{uid}.png"), show_layer_activations=True, show_shapes=True)

    # Initialize model callbacks
    checkpointer = ModelCheckpoint(
        os.path.join(chkpt_dir, f"ResNet101_{uid}.h5"), verbose=1, save_best_only=True
    )
    callbacks = [checkpointer]

    # Training
    results = model.fit(
                x=data_dict['train_images'],
                y=data_dict['train_masks'],
                batch_size=BATCH_SIZE,
                epochs=MAX_EPOCHS,
                validation_data=(data_dict['valid_images'], data_dict['valid_masks']),
                callbacks=callbacks,
                verbose=1,
                use_multiprocessing=True
    )

    # Evaluate performance on test set
    model.evaluate(data_dict['test_images'], data_dict['test_masks'])

    # TODO: Visualize train and validation metrices using result object

    # TODO: Predict for new data

    # TODO: Visualize prediction

if __name__ == "__main__":
    train()
    print("Training completed...")
