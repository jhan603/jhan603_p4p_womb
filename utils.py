import os
import albumentations as A
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

def list_files(folder, extension=".png", file_prefix=None, files_to_pick=None):
    """
    List files in a folder with specific filtering options.

    :param str folder: The folder to check for files.
    :param str extension: The file extension to filter by (default: ".png").
    :param str file_prefix: Optional prefix for file names (default: None).
    :param list[str] files_to_pick: List of specific file prefixes to pick (default: None).
    :return list[str]: List of selected file paths.
    """
    file_paths = []
    for root, dir, files in os.walk(folder):
        for file in files:
            if file.endswith(extension):
                if file_prefix is not None and not file.startswith(file_prefix):
                    continue

                if files_to_pick is not None and not any(os.path.splitext(file)[0].endswith(suffix) for suffix in files_to_pick):
                    continue

                file_paths.append(os.path.join(root, file))

    return file_paths

def list_images_with_masks(img_dir, mask_dir, file_prefix=None, files_to_pick=None):
    """
    List file paths of images and corresponding masks with optional filtering.

    :param str img_dir: Directory path containing image files.
    :param str mask_dir: Directory path containing mask files.
    :param str file_prefix: Optional prefix for file names (defaults to None).
    :param list[str] files_to_pick: List of specific files to pick (defaults to None).
    :return:
        - list[str]: List of file paths to image files.
        - list[str]: List of file paths to mask files.
    """
    image_path = list_files(img_dir, extension=".png", file_prefix=file_prefix, files_to_pick=files_to_pick)
    mask_path = list_files(mask_dir, extension=".png", file_prefix=file_prefix, files_to_pick=files_to_pick)
    return image_path, mask_path


def load_images_and_masks(image_dir, mask_dir, file_prefix=None, files_to_pick=None):
    """
    Load images and corresponding masks from specified directories.

    :param str image_dir: Directory path containing image files.
    :param str mask_dir: Directory path containing mask files.
    :param str file_prefix: Optional prefix for file names (defaults to None).
    :param list[str] files_to_pick: List of specific files to load (defaults to None).
    :return:
        - np.array: Array of loaded images.
        - np.array: Array of loaded masks.
    """
    image_paths, mask_paths = list_images_with_masks(image_dir, mask_dir, file_prefix, files_to_pick)
    images = load_images(image_paths)
    masks = load_images(mask_paths)
    return images, masks


def load_images(image_paths):
    """
    Load images from a list of image file paths.

    :param list[str] image_paths: List of file paths to the image files.
    :return: A numpy array containing the loaded images.
    :rtype np.ndarray: Array of loaded images.
    """
    images = np.zeros((len(image_paths), *imread(image_paths[0]).shape), dtype=np.uint8)
    for i, image_path in enumerate(image_paths):
        images[i] = imread(image_path)
    return images


def save_images(image_data, image_paths, save_folder):
    """
    Save images to the specified save folder.

    :param np.ndarray image_data: Array of image data to be saved.
    :param list[str] image_paths: List of source file paths for reference.
    :param str save_folder: Directory path where images will be saved.
    """
    os.makedirs(save_folder, exist_ok=True)
    for d, p in zip(image_data, image_paths):
        file_name = os.path.basename(p)
        file_path = os.path.join(save_folder, file_name)
        imsave(file_path, d.astype(np.uint8))


def display_image_and_mask(image, mask, i, save_only=False, figure_dir=None):
    """
    Display an image and its corresponding mask side by side.

    :param np.ndarray image: Image data to be displayed.
    :param np.ndarray mask: Mask data to be displayed.
    :param int i: Index or identifier for the image and mask (for title).
    :param bool save_only: If True, save the visualization as an image file (defaults to False).
    :param str figure_dir: Directory path where the visualization will be saved (if save_only is True).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Display image
    axes[0].imshow(image, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title(f'Image {i}')

    # Display mask
    axes[1].imshow(mask, cmap='viridis', alpha=0.5)
    axes[1].axis('off')
    axes[1].set_title(f'Mask {i}: Uniques {np.unique(mask)}')

    plt.tight_layout()

    if save_only and figure_dir is not None:
        plt.savefig(os.path.join(figure_dir, f"visualize_image_with_mask_{i}.png"))
    else:
        plt.show()


def display_transformation(images, masks, trans_images, trans_masks, i, save_only=False, figure_dir=None):
    """
    Display original and transformed images alongside their masks.

    :param np.ndarray images: Array of original images.
    :param np.ndarray masks: Array of original masks.
    :param np.ndarray trans_images: Array of transformed images.
    :param np.ndarray trans_masks: Array of transformed masks.
    :param int i: Index or identifier for the images and masks.
    :param bool save_only: If True, save the visualization as an image file (defaults to False).
    :param str figure_dir: Directory path where the visualization will be saved (if save_only is True).
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 18))

    # Display image
    axes[0, 0].imshow(trans_images[i], cmap='gray')
    axes[0, 0].axis('off')
    axes[0, 0].set_title(f'Image {i}')

    # Display mask
    axes[0, 1].imshow(trans_masks[i], cmap='viridis', alpha=0.5)
    axes[0, 1].axis('off')
    axes[0, 1].set_title(f'Mask {i}: Uniques are {np.unique(trans_masks[i])}')

    # Display image
    axes[1, 0].imshow(images[i], cmap='gray')
    axes[1, 0].axis('off')
    axes[1, 0].set_title(f'Image {i}')

    # Display mask
    axes[1, 1].imshow(masks[i], cmap='viridis', alpha=0.5)
    axes[1, 1].axis('off')
    axes[1, 1].set_title(f'Mask {i}: Uniques are {np.unique(masks[i])}')

    plt.tight_layout()

    if save_only and figure_dir is not None:
        plt.savefig(os.path.join(figure_dir, f"visualize_image_with_mask_{i}.png"))
    else:
        plt.show()


def transform_dataset(images, masks, input_shape=(256, 256)):
    """
    Transform a dataset of images and masks using augmentation techniques.

    :param np.ndarray images: Array of original images.
    :param np.ndarray masks: Array of original masks.
    :param tuple input_shape: Desired input shape for images after transformation (default: (256, 256)).
    :return:
        - np.ndarray: Array of transformed images.
        - np.ndarray: Array of transformed masks.
    """
    # Check more on the transformation functions in
    # https://albumentations.ai/docs/api_reference/full_reference/
    transform = A.Compose([
        A.Resize(*input_shape),
        A.Normalize(mean=0.0, std=1.0), # Rescale between 0 and 1
        A.HorizontalFlip(p=0.5) # Using a random flip
    ])

    transformed_images = []
    transformed_masks = []

    for i in range(images.shape[0]):
        augmented = transform(image=images[i], mask=masks[i])
        transformed_images.append(augmented['image'])
        transformed_masks.append(augmented['mask'])

    transformed_images = np.array(transformed_images)
    transformed_masks = np.array(transformed_masks)

    print(f"Original images: Min {np.min(images)} and Max {np.max(images)}")
    print(f"Transformed images: Min {np.min(transformed_images)} and Max {np.max(transformed_images)}")
    print(f"Transformed images: Shape {transformed_images.shape}")
    print(f"Unique values in transformed masks: {np.unique(transformed_masks)}")

    print("Mapping 254 and 255 to 4 and 5")
    transformed_masks[transformed_masks == 254] = 4
    transformed_masks[transformed_masks == 255] = 5
    print(f"Unique values in transformed masks: {np.unique(transformed_masks)}")

    # if using categorical cross entropy, then need to convert labels to one-hot-encoding
    # with sparce categorical cross entropy, it is not needed
    # transformed_masks = to_categorical(transformed_masks, num_classes=num_classes)

    # reshape only if using sparce categorical cross entropy
    # transformed_masks = transformed_masks.reshape(-1)
    print(f"Transformed masks reshaped: Shape {transformed_masks.shape}")
    return transformed_images, transformed_masks
