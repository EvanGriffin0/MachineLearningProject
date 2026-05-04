"""
transfer_model.py
-----------------
Transfer learning model builders for the image classification project.
"""

from tensorflow import keras


BASE_MODELS = {
    "mobilenetv2": keras.applications.MobileNetV2,
    "resnet50": keras.applications.ResNet50,
    "vgg16": keras.applications.VGG16,
}

PREPROCESS_INPUTS = {
    "mobilenetv2": keras.applications.mobilenet_v2.preprocess_input,
    "resnet50": keras.applications.resnet50.preprocess_input,
    "vgg16": keras.applications.vgg16.preprocess_input,
}


def build_transfer_model(
    input_shape=(224, 224, 3),
    num_classes=10,
    base_name="mobilenetv2",
    dense_units=128,
    dropout_rate=0.3,
    learning_rate=1e-3,
):
    """Build a frozen transfer-learning classifier."""
    model_key = base_name.lower()
    if model_key not in BASE_MODELS:
        supported = ", ".join(sorted(BASE_MODELS))
        raise ValueError(f"Unsupported base model '{base_name}'. Choose one of: {supported}")

    inputs = keras.Input(shape=input_shape)
    x = PREPROCESS_INPUTS[model_key](inputs)
    base_model = BASE_MODELS[model_key](
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    base_model.trainable = False

    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(dense_units, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"{model_key}_transfer")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def unfreeze_top_layers(model, base_model, trainable_layers=20, learning_rate=1e-5):
    """Unfreeze the top layers of the base model and recompile for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
