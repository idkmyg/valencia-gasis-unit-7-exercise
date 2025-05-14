import os

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    if not data:
        print(f"No content in file: {file_path}")
    return [line.strip() for line in data if line.strip()]

def split_data(data, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    if not abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Train, dev and test ratios must sum to 1.")
    # Shuffle data for randomness
    data = list(data)  # ensure it's a list in case it's an iterator
    from random import shuffle
    shuffle(data)
    total = len(data)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)
    
    train_set = data[:train_end]
    dev_set = data[train_end:dev_end]
    test_set = data[dev_end:]
    
    return train_set, dev_set, test_set