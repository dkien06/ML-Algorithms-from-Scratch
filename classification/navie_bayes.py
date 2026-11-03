
import re
from collections import Counter
import numpy as np


def email_to_words(email):
    email = email.lower()
    return re.findall(r"[a-z]+", email)


class NaiveBayesEmailClassification:

    def fit(self, spam_words, ham_words, total_email_spam, total_email_ham):
        if total_email_spam <= 0 or total_email_ham <= 0:
            raise ValueError("Both classes must contain at least one email.")

        # Count word frequencies
        self.spam_dict = Counter(spam_words)
        self.ham_dict = Counter(ham_words)

        self.total_spam_words = sum(self.spam_dict.values())
        self.total_ham_words = sum(self.ham_dict.values())

        # Vocabulary shared by both classes
        self.vocabulary = set(self.spam_dict) | set(self.ham_dict)
        self.vocab_size = len(self.vocabulary)

        # Class priors
        total_emails = total_email_spam + total_email_ham

        self.spam_rate = total_email_spam / total_emails
        self.ham_rate = total_email_ham / total_emails

        return self

    def calculate_likelihood(self, words):
        """
        Return log P(words | spam) and log P(words | ham).
        """
        log_likelihood = np.zeros(2)

        for word in words:
            spam_prob = (
                self.spam_dict.get(word, 0) + 1
            ) / (self.total_spam_words + self.vocab_size)

            ham_prob = (
                self.ham_dict.get(word, 0) + 1
            ) / (self.total_ham_words + self.vocab_size)

            log_likelihood[0] += np.log(spam_prob)
            log_likelihood[1] += np.log(ham_prob)

        return log_likelihood

    def choose_label(self, words):
        log_likelihood = self.calculate_likelihood(words)

        spam_score = np.log(self.spam_rate) + log_likelihood[0]
        ham_score = np.log(self.ham_rate) + log_likelihood[1]

        return spam_score > ham_score

    def predict(self, X):
        return [self.choose_label(words) for words in X]
