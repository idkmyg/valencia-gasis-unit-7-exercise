# Kneser–Ney–Smoothed N-gram Language Model

This project implements a Kneser–Ney–Smoothed N-gram Language Model using training data from the `wiki.train.tokens` file. The model is designed to predict the next word in a sequence based on the context of previous words.

## Project Structure

```
ngram-ml-project
├── data
│   └── wiki.train.tokens        # Training data for the model
├── src
│   ├── text_prediction.py       # Main entry point for loading data and making predictions
│   ├── kneser_ney.py            # Implementation of the Kneser–Ney-Smoothing algorithm
│   └── utils.py                 # Utility functions for data preprocessing
├── requirements.txt             # Python dependencies for the project
└── README.md                    # Documentation for the project
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your `wiki.train.tokens` file in the `data` directory.
2. Run the `text_prediction.py` script to load the training data and initialize the Kneser–Ney model:

```bash
python src/text_prediction.py
```

3. Follow the prompts to make predictions based on the trained model.

## Dependencies

This project requires the following Python libraries:

- NumPy
- pandas
- Any other libraries specified in `requirements.txt`

## License

This project is licensed under the MIT License. See the LICENSE file for more details.