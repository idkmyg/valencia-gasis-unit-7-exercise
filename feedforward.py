import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback

# 1. Load Penn TreeBank Dataset from local file "ptb.train.txt"
print("Loading Penn TreeBank dataset from local file...")
file_path = os.path.join(os.path.dirname(__file__), "ptb.train.txt")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset file not found: {file_path}")
with open(file_path, "r", encoding="utf-8") as f:
    texts = f.read().splitlines()

# 2. Preprocessing
print("Preprocessing dataset...")
combined_text = "\n".join([line.strip() for line in texts if line.strip()])
lines = combined_text.split("\n")

# Debug prints for preprocessing
print("Number of processed lines:", len(lines))

# Tokenize the text
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(lines)
total_words = len(tokenizer.word_index) + 1
print("Vocabulary size (total_words):", total_words)

# Convert text into sequences of tokens (n-gram sequences)
input_sequences = []
for line in lines:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[: i + 1]
        input_sequences.append(n_gram_sequence)

if not input_sequences:
    raise ValueError("No input sequences created. Check your dataset and preprocessing.")

# Pad sequences to make them uniform
max_seq_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding="pre")
print("Max sequence length:", max_seq_len)

# Split input and target (keeping y as integers for sparse categorical crossentropy)
X = input_sequences[:, :-1]  # Input sequence
y = input_sequences[:, -1]   # Target word (integer labels)

# 3. Split data into train and dev sets (80% train, 20% dev)
def split_dataset(X, y, train_ratio=0.8):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    train_end = int(len(X) * train_ratio)
    train_X, train_y = X[:train_end], y[:train_end]
    dev_X, dev_y = X[train_end:], y[train_end:]
    return (train_X, train_y), (dev_X, dev_y)

(train_X, train_y), (dev_X, dev_y) = split_dataset(X, y)
print(f"Dataset split: Train = {len(train_X)} samples, Dev = {len(dev_X)} samples")

# 4. Model Building (Feedforward Model)
print("Building the model...")
model = Sequential([
    Embedding(total_words, 100, input_length=max_seq_len - 1),
    GlobalAveragePooling1D(),
    Dense(150, activation="relu"),
    Dense(total_words, activation="softmax")
])
# Use sparse categorical crossentropy since y are integer labels
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()  # Should now show nonzero parameters if total_words > 0

# 5. Model Training with progress bar (short training run: 3 epochs, batch size increased)
print("Training the model...")
early_stopping = EarlyStopping(monitor="loss", patience=1, restore_best_weights=True)
tqdm_callback = TqdmCallback(verbose=1)  # in-place progress bar
history = model.fit(train_X, train_y,
                    epochs=3,
                    batch_size=128,
                    validation_data=(dev_X, dev_y),
                    callbacks=[early_stopping, tqdm_callback],
                    verbose=0)

# 6. Evaluation: Compute performance metrics on dev set including perplexity
print("\nEvaluating model on development data...")
loss, accuracy = model.evaluate(dev_X, dev_y, verbose=0)
perplexity = np.exp(loss)
print("\nPerformance Metrics:")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Perplexity: {perplexity:.4f}")

# 7. Text Generation
def generate_text(seed_text, next_words, max_seq_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding="pre")
        predicted = model.predict(token_list, verbose=0)
        predicted_index = tf.argmax(predicted[0]).numpy()
        predicted_word = tokenizer.index_word.get(predicted_index, "")
        seed_text += " " + predicted_word
    return seed_text

# Generate sample text
print("Generating text...")
seed_text = "The purpose of this"
generated_text = generate_text(seed_text, next_words=20, max_seq_len=max_seq_len)
print("Generated Text:", generated_text)