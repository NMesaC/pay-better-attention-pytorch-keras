"""
## "You Need to Pay Better Attention" Keras Transformer Example

## Paper Link: https://arxiv.org/abs/2403.01643

## Author: Nicholas Mesa-Cucalon (https://github.com/NMesaC)

## NOTE: This implementation has a learned W_a layer for EACH attention head
"""
import keras
import tensorflow as tf
from keras import ops
from keras import layers
import os
import tempfile
import time

from attention_indiv_wa import MultiHeadAttention

"""
## Transformer Block Module
"""
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, max_len, layer_type = 'SDPA', rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(n_heads=num_heads,
                                      d_model=embed_dim,
                                      d_k=embed_dim // num_heads,
                                      d_v=embed_dim // num_heads,
                                      max_len=max_len,
                                      layer_type=layer_type)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

"""
## Embedding Layer
"""
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

"""
## Model Size Helper
"""
def get_model_size(model):
    # Save the model to a temporary file
    _, keras_file = tempfile.mkstemp('.keras')
    model.save(keras_file, include_optimizer=True)

    # Get the file size
    size_bytes = os.path.getsize(keras_file)

    # Convert to MB
    size_mb = size_bytes / (1024 * 1024)

    # Delete the temporary file
    os.remove(keras_file)

    return size_mb

def main():
    # Setup initial variables
    vocab_size = 20000  # Only consider the top 20k words
    maxlen     = 32     # Only consider the first 32 words of each movie review
    embed_dim  = 32     # Embedding size for each token
    ff_dim     = 32     # Hidden layer size in feed forward network inside transformer
    batch_size = 64     # Batch Size
    epochs     = 10     # Number of epochs
    num_heads  = 4      # Number of attention heads
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
    print(len(x_train), "Training sequences")
    x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
    x_test = keras.utils.pad_sequences(x_test, maxlen=maxlen)

    # Train a Transformer Model with Each Attention Layer Type
    num_runs = 5
    layer_types = ['SDPA','Optimised','Efficient','Super']
    for layer in layer_types:
        avg_test_acc, avg_test_loss = 0, 0
        avg_run_time = 0
        for run in range(num_runs):
            print(f"Training with {layer} training layer")
            # Create inputs
            inputs = layers.Input(shape=(maxlen,))
            # Create model
            embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
            x = embedding_layer(inputs)
            transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, maxlen, layer)
            x = transformer_block(x)
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dropout(0.1)(x)
            x = layers.Dense(6, activation="relu")(x)
            x = layers.Dropout(0.1)(x)
            outputs = layers.Dense(1, activation="sigmoid")(x)

            # Initialize model
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer="adam", loss="BCE", metrics=["accuracy"]
            )

            # Create callbacks to save best model
            checkpoint_filepath = "./results/imdb/model/" + layer + f"_{num_heads}_heads" + "/run_num_" + str(run) + "/" + "model.weights.h5"
            os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
            # NOTE: Keras callback will save only the weights of the transformer model when val_accuracy goes up
            checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filepath,
                                                                  monitor="val_accuracy",
                                                                  save_best_only=True,
                                                                  save_weights_only=True,
                                                                 )
            # Fit model and retrieve history + time taken
            start_time = time.time()
            history = model.fit(x=x_train, 
                                y=y_train,
                                batch_size=batch_size, 
                                epochs=epochs, 
                                validation_split=0.1, 
                                callbacks=[checkpoint_callback])
            end_time = time.time()
            duration = end_time - start_time
            avg_run_time += duration
            print(f"Training took {duration:.4f} seconds")

            # Calculate model size
            model_size = get_model_size(model)
            print(f"The size of the model is approximately {model_size:.4f} MB")

            # Compute number of attention parameters
            num_of_attention_params = model.layers[2].att.count_params()
            print(f"Number of attention params: {num_of_attention_params}")

            # Compute and display test loss and accuracy for best model
            model.load_weights(checkpoint_filepath)
            test_loss, test_acc = model.evaluate(x_test, y_test)
            print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}\n")
            avg_test_loss += test_loss
            avg_test_acc  += test_acc
        avg_test_loss /= num_runs
        avg_test_acc  /= num_runs
        avg_run_time  /= num_runs
        print(f"Average Test Acc over {num_runs} for {layer}: {avg_test_acc} \n")
        print(f"Average Test Loss over {num_runs} for {layer}: {avg_test_loss} \n")
        print(f"Average Run Time over {num_runs} for {layer}: {avg_run_time} \n")
        print("\n")

if __name__ == '__main__':
    main()