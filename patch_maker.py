import os
import numpy as np
from skimage.io import imread, imsave
from skimage.util import view_as_blocks
from sklearn.model_selection import train_test_split
from utils import list_images_with_masks, load_images, save_images, display_image_and_mask

working_dir = os.getcwd()
                       # Root directory when working at university
image_dir = os.path.join(working_dir, "original_data", "images")
mask_dir = os.path.join(working_dir, "original_data", "masks")

figure_dir = os.path.join(working_dir, "figures")
os.makedirs(figure_dir, exist_ok=True)

has_labels = ["0000", "0001", "0002", "0003", "0004"]
file_prefix = "H"
# "Only slices 1-5, 10, 15, 20, 25, 30 have manual labels at this stage" was said in the email but slide 0014 doesnt have the appropriate mask.

def get_patches(image_paths, mask_paths, is_rgb, patch_size=(256, 256, 3)):
    """
    Extract and save patches from images and masks.

    :param list[str] image_paths: List of file paths to image files.
    :param list[str] mask_paths: List of file paths to mask files.
    :param bool is_rgb: Indicates whether the images are in RGB format (True) or grayscale (False).
    :param tuple patch_size: Size of the patches to extract (default: (256, 256, 3)).
    :return:
        - list[str]: List of file paths to the extracted image patches.
        - list[str]: List of file paths to the extracted mask patches.
    """
    patch_img_dir = os.path.join(working_dir, "patches", "images")
    patch_mask_dir = os.path.join(working_dir, "patches", "masks")

    os.makedirs(patch_img_dir, exist_ok=True)
    os.makedirs(patch_mask_dir, exist_ok=True)

    for image_path, mask_path in zip(image_paths, mask_paths):
        image = imread(image_path)
        mask = imread(mask_path)
        idx = int((image_path.split("_")[-1]).split(".")[0])

        display_image_and_mask(image, mask, idx, save_only=True, figure_dir=figure_dir)

        patches_image = view_as_blocks(image, patch_size)
        if is_rgb:
            patches_mask = view_as_blocks(mask, patch_size[:2])
        else:
            patches_mask = view_as_blocks(mask, patch_size)

        print(patches_image.shape)
        print(patches_mask.shape)

        for i in range(patches_image.shape[0]):
            for j in range(patches_image.shape[1]):
                patch_image = patches_image[i, j, 0, :, :, :]  # Extract RGB block
                patch_mask = patches_mask[i, j, :, :]  # Extract grayscale block

                # Save patches to appropriate folders
                image_patch_filename = os.path.basename(image_path).replace('.png', f'_patch_{i}_{j}.png')
                mask_patch_filename = os.path.basename(mask_path).replace('.png', f'_patch_{i}_{j}.png')

                imsave(os.path.join(patch_img_dir, image_patch_filename), patch_image)
                imsave(os.path.join(patch_mask_dir, mask_patch_filename), patch_mask)

    split_img_path, split_mask_path = list_images_with_masks(patch_img_dir, patch_mask_dir)

    return split_img_path, split_mask_path


def is_all_black(image):
    """
    Check if an image is completely black (contains only zeros).

    :param np.ndarray image: Input image as a NumPy array.
    :return bool: True if the image is completely black, False otherwise.
    """
    return np.all(image == 0)


def remove_all_black_imgs(image_paths, mask_paths):
    """
    Remove images that are completely black (contain only zeros) from the provided paths.

    :param list[str] image_paths: List of file paths to image files.
    :param list[str] mask_paths: List of file paths to mask files corresponding to the images.
    :return:
        - list[str]: List of file paths to non-black images.
        - list[str]: List of file paths to non-black masks corresponding to the non-black images.
    """
    non_black_image_paths = []
    non_black_mask_paths = []

    for image_path, mask_path in zip(image_paths, mask_paths):
        image = imread(image_path)
        if not is_all_black(image):
            non_black_image_paths.append(image_path)
            non_black_mask_paths.append(mask_path)
        else:  #remove all black image from disk
            os.remove(image_path)
            os.remove(mask_path)

    return non_black_image_paths, non_black_mask_paths


def split_and_save_data(image_paths, mask_paths, main_folder, valid_size, test_size):
    """
    Split and save a dataset into training, validation, and test sets.

    :param list[str] image_paths: List of file paths to image files.
    :param list[str] mask_paths: List of file paths to mask files corresponding to the images.
    :param str main_folder: Main folder where the data will be saved.
    :param float valid_size: Proportion of data to allocate for validation (0.0 to 1.0).
    :param float test_size: Proportion of data to allocate for testing (0.0 to 1.0).
    """
    print("Splitting data")
    images = load_images(image_paths).astype(np.int16)
    masks = load_images(mask_paths).astype(np.int16)
    assert len(images) == len(masks) == len(image_paths) == len(mask_paths)

    train_images, temp_images, train_masks, temp_masks, train_img_paths, temp_img_paths, train_mask_paths, temp_mask_paths = train_test_split(
        images, masks, image_paths, mask_paths, test_size=(valid_size + test_size), random_state=42, shuffle=True)

    valid_images, test_images, valid_masks, test_masks, valid_img_paths, test_img_paths, valid_mask_paths, test_mask_paths = train_test_split(
        temp_images, temp_masks, temp_img_paths, temp_mask_paths, test_size=test_size / (valid_size + test_size), random_state=42, shuffle=True)

    os.makedirs(main_folder, exist_ok=True)
    save_images(train_images, train_img_paths, os.path.join(main_folder, "train", "images"))
    save_images(valid_images, valid_img_paths, os.path.join(main_folder, "valid", "images"))
    save_images(test_images, test_img_paths, os.path.join(main_folder, "test", "images"))

    save_images(train_masks, train_mask_paths, os.path.join(main_folder, "train", "masks"))
    save_images(valid_masks, valid_mask_paths, os.path.join(main_folder, "valid", "masks"))
    save_images(test_masks, test_mask_paths, os.path.join(main_folder, "test", "masks"))

    print(f"Unique mask values in train set: {np.unique(train_masks)}")
    print(f"Unique mask values in valid set: {np.unique(valid_masks)}")
    print(f"Unique mask values in test set: {np.unique(test_masks)}")
    print(f"Min and Max values in train set: {np.max(train_images)} {np.min(train_images)}")
    print(f"Min and Max values in valid set: {np.max(valid_images)} {np.min(valid_images)}")
    print(f"Min and Max values in test set: {np.max(test_images)} {np.min(test_images)}")


if __name__ == "__main__":
    image_paths, mask_paths = list_images_with_masks(image_dir, mask_dir, file_prefix=file_prefix, files_to_pick=has_labels)

    tmp_img = imread(image_paths[0])
    print(f"Your image size is: {tmp_img.shape}")

    # TODO: Dynamic patch size
    # # Get a patch W and H
    # factor = 8
    # patch_w = tmp_img.shape[0]//factor
    # patch_h = tmp_img.shape[1]//factor

    # If RGB image patch size should be WxHx3
    patch_size = (118, 178, 3)
    print(f"Your patch size is: {patch_size}")

    split_img_path, split_mask_path = get_patches(image_paths, mask_paths, is_rgb=True, patch_size=patch_size)

    # TODO: Modify this to delete patches that have more than some percentage of black (Ex: more than 90% black)
    non_black_image_paths, non_black_mask_paths = remove_all_black_imgs(split_img_path, split_mask_path)

    split_and_save_data(non_black_image_paths, non_black_mask_paths, os.path.join(working_dir, "final_data"), valid_size=0.15, test_size=0.15)

    print("Done with data preprocessing")
