import collections
import math
from tqdm import tqdm
import pickle

class KneserNeyModel:
    def __init__(self, n=4, discount=0.5):
        self.n = n
        self.discount = discount
        self.ngram_counts = collections.defaultdict(int)
        self.context_counts = collections.defaultdict(int)
        self.unique_continuations = collections.defaultdict(set)
        self.unigram_counts = collections.defaultdict(int)
        self.total_unigrams = 0

    def train(self, corpus, epochs=1, progress_interval=100000):
        # Reset counts
        self.ngram_counts = collections.defaultdict(int)
        self.context_counts = collections.defaultdict(int)
        self.unique_continuations = collections.defaultdict(set)
        self.unigram_counts = collections.defaultdict(int)
        self.total_unigrams = 0

        total_sentences = len(corpus)
        overall_bar = tqdm(total=epochs * total_sentences, desc="Overall Training", unit="sentences", dynamic_ncols=True)
        for epoch in range(1, epochs + 1):
            tqdm.write(f"Starting epoch {epoch}/{epochs}...")
            epoch_bar = tqdm(total=total_sentences, desc=f"Epoch {epoch}", unit="sentence", leave=False, dynamic_ncols=True)
            for i, sentence in enumerate(corpus, start=1):
                tokens = sentence.split()
                # Pad with start and end tokens
                tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']

                # Update unigram counts (excluding the start tokens)
                for token in tokens[self.n - 1:]:
                    self.unigram_counts[token] += 1
                    self.total_unigrams += 1

                # Update n-gram counts and context counts
                for j in range(len(tokens) - self.n + 1):
                    ngram = tuple(tokens[j:j+self.n])
                    context = ngram[:-1]
                    word = ngram[-1]
                    self.ngram_counts[ngram] += 1
                    self.context_counts[context] += 1
                    self.unique_continuations[context].add(word)

                if i % progress_interval == 0:
                    epoch_bar.set_postfix(sentences_processed=i)
                epoch_bar.update(1)
                overall_bar.update(1)
            epoch_bar.close()
            tqdm.write(f"Completed epoch {epoch}/{epochs}.\n")
        overall_bar.close()

    def probability(self, context, word):
        """
        Recursive interpolated Kneser-Ney probability.
        context: tuple of tokens (length from 0 to n-1)
        """
        if len(context) == 0:
            # Base: unigram probability, ensure a floor value to avoid log(0)
            base_prob = (self.unigram_counts[word] / self.total_unigrams) if self.total_unigrams > 0 else 0
            return max(base_prob, 1e-10)

        c_context_word = self.ngram_counts.get(context + (word,), 0)
        c_context = self.context_counts.get(context, 0)
        if c_context > 0:
            lambda_val = (self.discount * len(self.unique_continuations[context])) / c_context
            p_lower = self.probability(context[1:], word)
            prob = max(c_context_word - self.discount, 0) / c_context + lambda_val * p_lower
            return max(prob, 1e-10)
        else:
            return self.probability(context[1:], word)

    def perplexity(self, corpus):
        total_log_prob = 0.0
        total_words = 0
        for sentence in corpus:
            tokens = sentence.split()
            tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
            N = len(tokens) - (self.n - 1)
            total_words += N
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i+self.n-1])
                word = tokens[i+self.n-1]
                prob = self.probability(context, word)
                total_log_prob += math.log(prob)
        avg_log_prob = total_log_prob / total_words
        return math.exp(-avg_log_prob)

    def predict(self, context, num_predictions=5):
        tokens = context.split()
        context_tuple = tuple(tokens[-(self.n - 1):]) if len(tokens) >= (self.n - 1) else tuple(tokens)
        candidates = set()
        for ngram in self.ngram_counts.keys():
            if len(context_tuple) > 0:
                if ngram[:-1][-len(context_tuple):] == context_tuple:
                    candidates.add(ngram[-1])
            else:
                candidates.add(ngram[-1])
        if not candidates:
            candidates = self.unigram_counts.keys()

        scores = {}
        for word in candidates:
            scores[word] = self.probability(context_tuple, word)
        predictions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in predictions[:num_predictions]]

    def save(self, filename):
        """
        Save the model to a file using pickle.
        """
        with open(filename, "wb") as f:
            pickle.dump({
                'n': self.n,
                'discount': self.discount,
                'ngram_counts': dict(self.ngram_counts),
                'context_counts': dict(self.context_counts),
                'unique_continuations': {k: list(v) for k, v in self.unique_continuations.items()},
                'unigram_counts': dict(self.unigram_counts),
                'total_unigrams': self.total_unigrams
            }, f)
        print(f"Model saved to {filename}")

    @staticmethod
    def load(filename):
        """
        Load a model saved with the save method.
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
        model = KneserNeyModel(n=data['n'], discount=data['discount'])
        model.ngram_counts = collections.defaultdict(int, data['ngram_counts'])
        model.context_counts = collections.defaultdict(int, data['context_counts'])
        # Convert lists back to sets for unique_continuations
        model.unique_continuations = collections.defaultdict(set, {k: set(v) for k, v in data['unique_continuations'].items()})
        model.unigram_counts = collections.defaultdict(int, data['unigram_counts'])
        model.total_unigrams = data['total_unigrams']
        return model