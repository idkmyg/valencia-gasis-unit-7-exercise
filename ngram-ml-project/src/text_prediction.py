import os
import tkinter as tk
from tkinter import filedialog
from kneser_ney import KneserNeyModel
from utils import load_data, split_data

def choose_file_to_load():
    root = tk.Tk()
    root.withdraw()  # Hide the main window.
    filename = filedialog.askopenfilename(
        title="Select a model file",
        filetypes=(("Pickle Files", "*.pkl"), ("All Files", "*.*"))
    )
    return filename

def choose_file_to_save():
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.asksaveasfilename(
        title="Save the model as",
        defaultextension=".pkl",
        filetypes=(("Pickle Files", "*.pkl"), ("All Files", "*.*"))
    )
    return filename

def train_model():
    # Load data and train model function
    train_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ptb.train.txt')
    data = load_data(train_file_path)
    train_set, _, _ = split_data(data, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1)
    kn_model = KneserNeyModel()
    kn_model.train(train_set, epochs=3, progress_interval=100000)
    return kn_model

def main():
    # Ask if user wants to load an existing model via a file picker
    load_choice = input("Do you want to load an existing model? (y/n): ")
    if load_choice.strip().lower() == 'y':
        filename = choose_file_to_load()
        if filename and os.path.exists(filename):
            kn_model = KneserNeyModel.load(filename)
            print(f"Model loaded from {filename}\n")
        else:
            print("No file was selected or file not found. Proceeding with training a new model...\n")
            kn_model = train_model()
    else:
        kn_model = train_model()
    
    # Evaluate performance on train, dev, and test sets
    train_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ptb.train.txt')
    data = load_data(train_file_path)
    train_set, dev_set, test_set = split_data(data, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1)
    print(f"Data split sizes -> Train: {len(train_set)}, Dev: {len(dev_set)}, Test: {len(test_set)}\n")
    
    train_pp = kn_model.perplexity(train_set)
    dev_pp = kn_model.perplexity(dev_set)
    test_pp = kn_model.perplexity(test_set)
    print(f"Perplexity -> Train: {train_pp:.2f}, Dev: {dev_pp:.2f}, Test: {test_pp:.2f}\n")
    
    # Example prediction
    context = "The quick brown"
    prediction = kn_model.predict(context)
    print(f"Predicted next words for '{context}': {prediction}")
    
    # Option to save the newly trained model if applicable
    if load_choice.strip().lower() != 'y':
        save_choice = input("Do you want to save the newly trained model? (y/n): ")
        if save_choice.strip().lower() == 'y':
            filename = choose_file_to_save()
            if filename:
                kn_model.save(filename)
    
if __name__ == "__main__":
    main()