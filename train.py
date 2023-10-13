import os
import datetime
import numpy as np
from model import *
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model

import matplotlib.pyplot as plt

from utils import load_images_and_masks, transform_dataset, display_transformation

# Input shape must be a factor of 2 and W==H.
# If want to try a shape that is not factor of 2 or W!=H, change the model accordingly
# in the upsampling section
INPUT_SHAPE = (256, 256, 3)
BATCH_SIZE = 16

#MAX_EPOCHS = 3    
#MAX_EPOCHS = 15
MAX_EPOCHS = 5


NUM_CLASSES = 5
# Figure out wat EPOCHS and BATCH_SIZE to give best performance

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

def train(model_name):

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
    print(f"{np.unique(data_dict['train_masks'])}")
    print(f"{np.unique(data_dict['test_masks'])}")
    print(f"{np.unique(data_dict['valid_masks'])}")

    # Initialize model

    if  model_name == "ResNet152V2": # Trash 
        model = create_ResNet152V2_segmentation_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, layer_no=0, weights='imagenet', include_top=False)
    elif model_name == "DenseNet121":
        model = create_densenet121_segmentation_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, layer_no=0, weights='imagenet', include_top=False)
    elif model_name == "VGG19":
        model = create_VGG19_segmentation_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, layer_no=0, weights='imagenet', include_top=False)
    elif model_name == "VGG16": # THis one was not that good too. 
        model = create_VGG16_segmentation_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, layer_no=0, weights='imagenet', include_top=False)
    elif model_name == "NASNetMobile": # Trash I think
        model = create_NASNetMobile_segmentation_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, layer_no=0, weights='imagenet', include_top=False)
    elif model_name == "DenseNet201":
        model = create_DenseNet201_segmentation_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, layer_no=0, weights='imagenet', include_top=False)
    elif model_name == "MobileNetV2":
        model = create_MobileNetV2_segmentation_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, layer_no=0, weights='imagenet', include_top=False)
    uid = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"Unique Identifier: {uid}")
    
    plot_model(model, to_file=os.path.join(figure_dir, f"{model_name}_{uid}.png"), show_layer_activations=True, show_shapes=True)

        # Initialize model callbacks
    checkpointer = ModelCheckpoint(
        os.path.join(chkpt_dir, f"{model_name}_{uid}.h5"), verbose=1, save_best_only=True
        )

    # Define the EarlyStopping callback to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=4,          # Stop training if no improvement for 4 epochs
        restore_best_weights=True  # Restore best weights when early stopping
    )   

    callbacks = [checkpointer, early_stopping]

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
    score = model.evaluate(data_dict['test_images'], data_dict['test_masks'])
    print(f"The Model {model_name} has an accuracy of {score[1]}, and a loss of {score[0]}")

    # TODO: Visualize train and validation metrices using result object

    # Get training and validation loss values from the 'results' object
    train_loss = results.history['loss']
    val_loss = results.history['val_loss']
    loss_scores = [train_loss, val_loss]

    # Get training and validation accuracy values (if available)
    if 'accuracy' in results.history:
        train_acc = results.history['accuracy']
        val_acc = results.history['val_accuracy']

        epochs = list(range(1, len(train_loss) + 1))

        # Plot training and validation accuracy
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_acc, label='Training Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.ylim(0,1)
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot training and validation loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Show the plots
        plt.tight_layout()
        Images_set_dir = os.path.join(working_dir, "Images_Sets")
        os.makedirs(Images_set_dir, exist_ok=True)
        plt.savefig(f"{Images_set_dir}\\{model_name}_Results.png")
        plt.close()
    

    return data_dict, model_name, model, score, loss_scores
    # TODO: Predict for new data
    # Do u 
def Predict_and_Display_Images(data_dict, model):
    print("Predicting Data...")

    test_images = data_dict['test_images']  # Load test images
    predictions = model.predict(test_images, verbose=1)

    print(f"Shape of predictions variable: {predictions.shape}")

    pred = np.argmax(predictions, axis = 3)
    # 0 -> 1
    # 1 -> 2
    # 2 -> 3
    # 3 -> 254
    # 4 -> 255

    pred[pred == 4] = 255
    pred[pred == 3] = 254
    pred[pred == 2] = 3
    pred[pred == 1] = 2
    pred[pred == 0] = 1

    #pred = np.expand_dims(pred, axis = -1)
    print(f"Shape of pred: {pred.shape}")

    print("Predictions completed...")

    # TODO: Visualize prediction

    ground_truth_masks = data_dict['test_masks']  # Load ground truth masks
    ground_truth_masks = np.squeeze(ground_truth_masks, axis = -1)

    ground_truth_masks[ground_truth_masks == 4] = 255
    ground_truth_masks[ground_truth_masks == 3] = 254
    ground_truth_masks[ground_truth_masks == 2] = 3
    ground_truth_masks[ground_truth_masks == 1] = 2
    ground_truth_masks[ground_truth_masks == 0] = 1

    print(f"Shape of ground_truth_masks variable: {ground_truth_masks.shape}")
    print(f"Unique values in transformed masks: {np.unique(ground_truth_masks)}")

    # TODO: Sample Compare image to mask and ground-truth mask

    len_test_imgs = test_images.shape[0]
    num_rand_imgs = 10
    random_indices = np.random.choice(len_test_imgs, num_rand_imgs, replace= False)

    t_img_comp = test_images[random_indices]
    pred_img_comp = pred[random_indices]
    g_truth_comp = ground_truth_masks[random_indices]

    # 13/09 TODO : Use the np.vectorize thing to map the values to a tuple?
    color_map = {
        1: 1,
        2: 75,
        3: 125,
        254: 175,
       255: 255
    }
    # print("Hello World)
    vectorized_map = np.vectorize(lambda value: color_map.get(value, 0))
    colored_truth = vectorized_map(g_truth_comp)
    colored_pred = vectorized_map(pred_img_comp)

    # Figure directory to save figures
    Images_set_dir = os.path.join(working_dir, "Images_Sets")
    os.makedirs(Images_set_dir, exist_ok=True)
    save_image = True

    for i in range(0,num_rand_imgs):
        plt.figure()

        plt.subplot(1,3,1)
        plt.imshow(t_img_comp[i], vmin = 1, vmax = 255)
        plt.title("Input Image")

        plt.subplot(1,3,2)
        plt.imshow(colored_truth[i], vmin = 1, vmax = 255)
        plt.title("Ground Truth")
        plt.colorbar()

        plt.subplot(1,3,3)
        plt.imshow(colored_pred[i], vmin = 1, vmax = 255)
        plt.title("Predicted Image")
        plt.colorbar()

        plt.tight_layout()

        # If True save the images, else just show it and don't save it. 
        if save_image:
            plt.savefig(f"{Images_set_dir}\\Image_Set_{i+1}.png")
            plt.close()
        else:
            plt.show()
    
    print("Visualised Predictions...")

if __name__ == "__main__":
    print("Initialize Training...")
    #models = ["ResNet152V2", "DenseNet121", "VGG19", "VGG16", "NASNetMobile", "DenseNet201", "MobileNetV2"]
    models = ["DenseNet201"]
    model_results = []
    for model_name in models:
        print(f"Training data on {model_name} Model")
        data_dict, model_name, model, score, loss_scores = train(model_name)
        model_results.append([model_name, loss_scores, model])
    
    print("Training completed...")
    Best_Model = min(model_results, key = lambda x: x[1][1])
    model_name, loss_scores, model = Best_Model
    print(f"Best Model is {model_name}, with an val_loss of {min(loss_scores[1])}")
    
    print(f"Saving model as a .keras file...")
    model.save("my_model.keras")
    # When loading in model
    # reconstructed_model = load_model("my_model.keras")

    Predict_and_Display_Images(data_dict, model)
    
    # TODO: Train and predict data with multiple different models
    # maybe look into like 5 or so. 
    # Return the best model
    # Train and eval based on model.evaluate
    # Then make prediction and visualize the masks
    # Write a plan when I start.

    # TODO: 12/09
    # 1. Create 3 more different models DONE
    # 2. Run Train Function on the 5 different models. 3 selected will be VGG19, MobileNetV2, EfficientNetV2S
    #    Return score, model, model_name and data_dict. 
    # 3. Best Eval Score, and using that model make predictions
    # 4. Visualize The Images. 

    # TODO: 13/09
    # Change the really crap models and try find better ones? DONE

    # Increase the number of epochs and see if that improves the models
    # looking at MAX_EPOCHS = 10 this doesn't seem to be the case DONE

    # Create a manual colour mapping
    # change values to hve better distribution DONE
    # Create a legend/thing that will easily identify the colours and what region they are. 

    # 14/09 Figure out good epoch too much, could mea over fitting
    # I THINK 10 epochs is good, implemented dropout in my model
    # TODO : 
    # Implement optimizing learning rate
    # Early stopping with patience = 2
    # Applied more ways to augment the images. 
    # Lowered batch size

    # Change how I pick the optimal model
    # Currently picked model with best accuracy, but I should look at valid_loss
    # to see how well the model when unseen data is fed into it. 
    # But yea my model is pretty crap atm still. 
    # Perhaps ask Alys for more images and masks
    # Ask how I can determine these blood vessels. 

    # Try improve on my learning rate. 
    # In my report talk about how shit some of the models were and had to replace them. 


