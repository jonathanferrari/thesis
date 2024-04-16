import tensorflow as tf
import numpy as np
import keras
import os
from keras import layers, ops
keras.config.disable_traceback_filtering()
from data import VideoDataGenerator, labels, files, labels_df
from keras.callbacks import ModelCheckpoint, CSVLogger


classes = list(labels_df.groupby('label').size().sort_values(ascending=False)[:10].index)
test_files = [f for f in files if labels[int(os.path.basename(f).split('.')[0])] in classes]


BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (20, 64, 64, 1)
# OPTIMIZER
NUM_CLASSES = len(labels_df['label'].unique())
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 10

# TUBELET EMBEDDING
PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 156
NUM_HEADS = 12
NUM_LAYERS = 12
train_gen = VideoDataGenerator(test_files, batch_size=32)

class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches
    
    def build(self, input_shape):
        self.projection.build(input_shape)
        self.flatten.build(self.projection.compute_output_shape(input_shape))
        super().build(input_shape)
class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = ops.arange(0, num_tokens, 1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens
    
def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    num_classes,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=ops.gelu),
                layers.Dense(units=embed_dim, activation=ops.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    outputs = layers.Dense(units=num_classes, activation="softmax")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def run_experiment(name, data = train_gen, validation = None, projection = PROJECTION_DIM, patch = PATCH_SIZE, learning_rate = LEARNING_RATE, epochs = EPOCHS):
    
    csv_logger = CSVLogger(f"../models/{name}.csv")
    checkpoint = ModelCheckpoint(f"../models/{name}.keras", save_best_only=True)
    
    model = create_vivit_classifier(
        tubelet_embedder=TubeletEmbedding(embed_dim=projection, patch_size=patch),
        positional_encoder=PositionalEncoder(embed_dim=projection),
        num_classes=NUM_CLASSES,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy"), keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy")],
    )
    if validation is not None:
        model.fit(data, validation_data=validation, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[csv_logger, checkpoint])
    else:
        model.fit(data, batch_size=BATCH_SIZE, epochs=epochs)
    _, accuracy, top_5_accuracy = model.evaluate(train_gen, batch_size=BATCH_SIZE)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Top 5 Accuracy: {top_5_accuracy * 100:.2f}%")
    
    return model