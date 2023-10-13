from keras.applications import ResNet101, DenseNet121, VGG19, MobileNetV2, ResNet152V2, NASNetLarge, EfficientNetV2L, VGG16, NASNetMobile, DenseNet201
from keras.models import Model
from keras.layers import UpSampling2D, Dropout, Conv2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import math


def set_trainable_layers(base_model, layer_no, structure = 'block'):
    """Fine-tune specific layers in a model based on layer number and structure.

    :param keras.model base_model: Pre-existing model to fine-tune.
    :param int/bool layer_no: Number of layers to make trainable or True for all, 0 or False for none.
    :param str structure: Structure keyword to filter layers (default: 'block').
    :return keras.models.Model: Modified model with trainable layers specified.
    """

    # if the number of layers_no is 0 or false then freeze the base model
    if layer_no == 0 or layer_no is False:
        base_model.trainable = False
        return base_model
    # if the number of layers_no is true then unfreeze the base model
    elif layer_no is True:
        base_model.trainable = True
        return base_model

    # double-check if layer_no is an int and
    # define a list of layers to consider for training
    if isinstance(layer_no, int):
        layers_to_train = base_model.layers[-layer_no:]

    # filter layers based on the specified structure
    trainable_layers = [layer for layer in layers_to_train if structure in layer.name]

    # set the trainable property for each layer in trainable
    for layer in base_model.layers:
        layer.trainable = layer in trainable_layers

    return base_model


def create_resnet101_segmentation_model(input_shape, num_classes, layer_no=0, weights='imagenet', include_top=True):
    """
    Create a ResNet101-based semantic segmentation model.

    :param tuple input_shape: Input shape of the model (e.g., (256, 256, 3)).
    :param int num_classes: Number of output classes for segmentation.
    :param int layer_no: Number of layers to modify (default: 0, which means no layers are modified).
    :param str weights: Specifies the pre-trained weights to use (default: 'imagenet').
    :param bool include_top: Whether to include the top (output) layer of the ResNet model (default: False).
    :return keras.models.Model: Compiled ResNet101-based segmentation model.
    """
    # Load the pretrained model
    base_model = ResNet101(
        input_shape=input_shape,
        weights=weights,
        include_top=include_top)

    # Set trainability
    base_model = set_trainable_layers(base_model, layer_no)

    # Create the decoder network
    x = base_model.output
    num_blocks = int(math.log2(input_shape[0]//x.shape[1]))

    # Upsample iteratively until input shape is reached
    for _ in range(num_blocks):
        num_filters = x.shape[-1] // 2
        x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.1)(x)
        x = UpSampling2D(size=(2, 2))(x)


    # Final output layer with 5 classes and softmax activation
    # x = Flatten()(x)
    # output = Dense(num_classes, activation='softmax')(x)
    output = Conv2D(num_classes, (1, 1), activation='softmax')(x)


    # Create the full segmentation model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    return model

def create_densenet121_segmentation_model(input_shape, num_classes, layer_no=0, weights='imagenet', include_top=False):
    """
    Create a Densenet121-based semantic segmentation model.

    :param tuple input_shape: Input shape of the model (e.g., (256, 256, 3)).
    :param int num_classes: Number of output classes for segmentation.
    :param int layer_no: Number of layers to modify (default: 0, which means no layers are modified).
    :param str weights: Specifies the pre-trained weights to use (default: 'imagenet').
    :param bool include_top: Whether to include the top (output) layer of the DenseNet model (default: False).
    :return keras.models.Model: Compiled Densenet121-based segmentation model.
    """
    # Load the pretrained model
    base_model = DenseNet121(
        input_shape=input_shape,
        weights=weights,
        include_top=include_top)

    # Set trainability
    base_model = set_trainable_layers(base_model, layer_no)

    # Create the decoder network
    x = base_model.output
    num_blocks = int(math.log2(input_shape[0]//x.shape[1]))

    # Upsample iteratively until input shape is reached
    for _ in range(num_blocks):
        num_filters = x.shape[-1] // 2
        x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.1)(x)
        x = UpSampling2D(size=(2, 2))(x)


    # Final output layer with 5 classes and softmax activation
    output = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    # Create the full segmentation model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    return model

def create_VGG19_segmentation_model(input_shape, num_classes, layer_no=0, weights='imagenet', include_top=False):
    """
    Create a VGG19-based semantic segmentation model.

    :param tuple input_shape: Input shape of the model (e.g., (256, 256, 3)).
    :param int num_classes: Number of output classes for segmentation.
    :param int layer_no: Number of layers to modify (default: 0, which means no layers are modified).
    :param str weights: Specifies the pre-trained weights to use (default: 'imagenet').
    :param bool include_top: Whether to include the top (output) layer of the VGG19 model (default: False).
    :return keras.models.Model: Compiled VGG19-based segmentation model.
    """
    # Load the pretrained model
    base_model = VGG19(
        input_shape=input_shape,
        weights=weights,
        include_top=include_top)

    # Set trainability
    base_model = set_trainable_layers(base_model, layer_no)

    # Create the decoder network
    x = base_model.output
    num_blocks = int(math.log2(input_shape[0]//x.shape[1]))

    # Upsample iteratively until input shape is reached
    for _ in range(num_blocks):
        num_filters = x.shape[-1] // 2
        x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.1)(x)
        x = UpSampling2D(size=(2, 2))(x)


    # Final output layer with 5 classes and softmax activation
    output = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    # Create the full segmentation model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    return model

def create_MobileNetV2_segmentation_model(input_shape, num_classes, layer_no=0, weights='imagenet', include_top=False):
    """
    Create a MobileNetV2-based semantic segmentation model.

    :param tuple input_shape: Input shape of the model (e.g., (256, 256, 3)).
    :param int num_classes: Number of output classes for segmentation.
    :param int layer_no: Number of layers to modify (default: 0, which means no layers are modified).
    :param str weights: Specifies the pre-trained weights to use (default: 'imagenet').
    :param bool include_top: Whether to include the top (output) layer of the MobileNetV2 model (default: False).
    :return keras.models.Model: Compiled MobileNetV2-based segmentation model.
    """
    # Load the pretrained model
    base_model = MobileNetV2(
        input_shape=input_shape,
        weights=weights,
        include_top=include_top)

    # Set trainability
    base_model = set_trainable_layers(base_model, layer_no)

    # Create the decoder network
    x = base_model.output
    num_blocks = int(math.log2(input_shape[0]//x.shape[1]))

    # Upsample iteratively until input shape is reached
    for _ in range(num_blocks):
        num_filters = x.shape[-1] // 2
        x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.1)(x)
        x = UpSampling2D(size=(2, 2))(x)


    # Final output layer with 5 classes and softmax activation
    output = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    # Create the full segmentation model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    return model

def create_VGG16_segmentation_model(input_shape, num_classes, layer_no=0, weights='imagenet', include_top=False):
    """
    Create a VGG16-based semantic segmentation model.

    :param tuple input_shape: Input shape of the model (e.g., (256, 256, 3)).
    :param int num_classes: Number of output classes for segmentation.
    :param int layer_no: Number of layers to modify (default: 0, which means no layers are modified).
    :param str weights: Specifies the pre-trained weights to use (default: 'imagenet').
    :param bool include_top: Whether to include the top (output) layer of the EfficientNetV2S model (default: False).
    :return keras.models.Model: Compiled VGG16-based segmentation model.
    """
    # Load the pretrained model
    base_model = VGG16(
        input_shape=input_shape,
        weights=weights,
        include_top=include_top)

    # Set trainability
    base_model = set_trainable_layers(base_model, layer_no)

    # Create the decoder network
    x = base_model.output
    num_blocks = int(math.log2(input_shape[0]//x.shape[1]))

    # Upsample iteratively until input shape is reached
    for _ in range(num_blocks):
        num_filters = x.shape[-1] // 2
        x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.1)(x)
        x = UpSampling2D(size=(2, 2))(x)


    # Final output layer with 5 classes and softmax activation
    output = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    # Create the full segmentation model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    return model

def create_ResNet152V2_segmentation_model(input_shape, num_classes, layer_no=0, weights='imagenet', include_top=False):
    """
    Create a ResNet152V2-based semantic segmentation model.

    :param tuple input_shape: Input shape of the model (e.g., (256, 256, 3)).
    :param int num_classes: Number of output classes for segmentation.
    :param int layer_no: Number of layers to modify (default: 0, which means no layers are modified).
    :param str weights: Specifies the pre-trained weights to use (default: 'imagenet').
    :param bool include_top: Whether to include the top (output) layer of the DenseNet model (default: False).
    :return keras.models.Model: Compiled ResNet152V2-based segmentation model.
    """
    # Load the pretrained model
    base_model = ResNet152V2(
        input_shape=input_shape,
        weights=weights,
        include_top=include_top)

    # Set trainability
    base_model = set_trainable_layers(base_model, layer_no)

    # Create the decoder network
    x = base_model.output
    num_blocks = int(math.log2(input_shape[0]//x.shape[1]))

    # Upsample iteratively until input shape is reached
    for _ in range(num_blocks):
        num_filters = x.shape[-1] // 2
        x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.1)(x)
        x = UpSampling2D(size=(2, 2))(x)


    # Final output layer with 5 classes and softmax activation
    output = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    # Create the full segmentation model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    return model

def create_NASNetLarge_segmentation_model(input_shape, num_classes, layer_no=0, weights='imagenet', include_top=False):
    """
    Create a NASNetLarge-based semantic segmentation model.

    :param tuple input_shape: Input shape of the model (e.g., (256, 256, 3)).
    :param int num_classes: Number of output classes for segmentation.
    :param int layer_no: Number of layers to modify (default: 0, which means no layers are modified).
    :param str weights: Specifies the pre-trained weights to use (default: 'imagenet').
    :param bool include_top: Whether to include the top (output) layer of the DenseNet model (default: False).
    :return keras.models.Model: Compiled NASNetLarge-based segmentation model.
    """
    # Load the pretrained model
    base_model = NASNetLarge(
        input_shape=input_shape,
        weights=weights,
        include_top=include_top)

    # Set trainability
    base_model = set_trainable_layers(base_model, layer_no)

    # Create the decoder network
    x = base_model.output
    num_blocks = int(math.log2(input_shape[0]//x.shape[1]))

    # Upsample iteratively until input shape is reached
    for _ in range(num_blocks):
        num_filters = x.shape[-1] // 2
        x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.1)(x)
        x = UpSampling2D(size=(2, 2))(x)


    # Final output layer with 5 classes and softmax activation
    output = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    # Create the full segmentation model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def create_EfficientNetV2L_segmentation_model(input_shape, num_classes, layer_no=0, weights='imagenet', include_top=False):
    """
    Create a EfficientNetV2L-based semantic segmentation model.

    :param tuple input_shape: Input shape of the model (e.g., (256, 256, 3)).
    :param int num_classes: Number of output classes for segmentation.
    :param int layer_no: Number of layers to modify (default: 0, which means no layers are modified).
    :param str weights: Specifies the pre-trained weights to use (default: 'imagenet').
    :param bool include_top: Whether to include the top (output) layer of the DenseNet model (default: False).
    :return keras.models.Model: Compiled EfficientNetV2L-based segmentation model.
    """
    # Load the pretrained model
    base_model = EfficientNetV2L(
        input_shape=input_shape,
        weights=weights,
        include_top=include_top)

    # Set trainability
    base_model = set_trainable_layers(base_model, layer_no)

    # Create the decoder network
    x = base_model.output
    num_blocks = int(math.log2(input_shape[0]//x.shape[1]))

    # Upsample iteratively until input shape is reached
    for _ in range(num_blocks):
        num_filters = x.shape[-1] // 2
        x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.1)(x)
        x = UpSampling2D(size=(2, 2))(x)


    # Final output layer with 5 classes and softmax activation
    output = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    # Create the full segmentation model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    return model

def create_NASNetMobile_segmentation_model(input_shape, num_classes, layer_no=0, weights='imagenet', include_top=False):
    """
    Create a NASNetMobile-based semantic segmentation model.

    :param tuple input_shape: Input shape of the model (e.g., (256, 256, 3)).
    :param int num_classes: Number of output classes for segmentation.
    :param int layer_no: Number of layers to modify (default: 0, which means no layers are modified).
    :param str weights: Specifies the pre-trained weights to use (default: 'imagenet').
    :param bool include_top: Whether to include the top (output) layer of the DenseNet model (default: False).
    :return keras.models.Model: Compiled NASNetMobile-based segmentation model.
    """
    # Load the pretrained model
    base_model = NASNetMobile(
        input_shape=input_shape,
        weights=weights,
        include_top=include_top)

    # Set trainability
    base_model = set_trainable_layers(base_model, layer_no)

    # Create the decoder network
    x = base_model.output
    num_blocks = int(math.log2(input_shape[0]//x.shape[1]))

    # Upsample iteratively until input shape is reached
    for _ in range(num_blocks):
        num_filters = x.shape[-1] // 2
        x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.1)(x)
        x = UpSampling2D(size=(2, 2))(x)


    # Final output layer with 5 classes and softmax activation
    output = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    # Create the full segmentation model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    return model

def create_DenseNet201_segmentation_model(input_shape, num_classes, layer_no=0, weights='imagenet', include_top=False):
    """
    Create a DenseNet201-based semantic segmentation model.

    :param tuple input_shape: Input shape of the model (e.g., (256, 256, 3)).
    :param int num_classes: Number of output classes for segmentation.
    :param int layer_no: Number of layers to modify (default: 0, which means no layers are modified).
    :param str weights: Specifies the pre-trained weights to use (default: 'imagenet').
    :param bool include_top: Whether to include the top (output) layer of the DenseNet model (default: False).
    :return keras.models.Model: Compiled DenseNet201-based segmentation model.
    """
    # Load the pretrained model
    base_model = DenseNet201(
        input_shape=input_shape,
        weights=weights,
        include_top=include_top)

    # Set trainability
    base_model = set_trainable_layers(base_model, layer_no)

    # Create the decoder network
    x = base_model.output
    num_blocks = int(math.log2(input_shape[0]//x.shape[1]))

    # Upsample iteratively until input shape is reached
    for _ in range(num_blocks):
        num_filters = x.shape[-1] // 2
        x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.1)(x)
        x = UpSampling2D(size=(2, 2))(x)


    # Final output layer with 5 classes and softmax activation
    output = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    # Create the full segmentation model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    return model