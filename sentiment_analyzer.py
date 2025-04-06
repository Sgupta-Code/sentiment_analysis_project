import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class SentimentAnalyzer:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.model = None

    def preprocess_text(self, text):
        """More advanced text preprocessing for better sentiment analysis results"""
        # Handle NaN values
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Replace contractions
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'m", " am", text)

        # Keep emoticons and sentiment-relevant punctuation
        emoticons = r"(?:[:=;][oO\-]?[D\)\]\(\]/\\OpP])"
        text = re.sub(r"[^a-zA-Z\!\?\.\,\s]", "", text)

        # Handle negations (very important for sentiment)
        # Add special token for "not" + word
        words = text.split()
        negation = False
        result = []
        negation_words = ['not', 'no', 'never', "n't", "cannot"]

        for word in words:
            if word in negation_words:
                negation = True
                result.append(word)
                continue
            if negation:
                result.append("NOT_" + word)
                if word in ['.', '!', '?']:
                    negation = False
            else:
                result.append(word)

        text = ' '.join(result)

        # Tokenize
        tokens = nltk.word_tokenize(text)

        # Remove stopwords (but keep negation words)
        stop_words_except_negations = {word for word in self.stop_words
                                       if word not in ['no', 'not', 'nor', 'neither']}

        # Apply stemming but preserve important sentiment words
        sentiment_words = {"love", "hate", "good", "bad", "best", "worst",
                           "great", "terrible", "excellent", "poor", "amazing"}

        cleaned_tokens = []
        for word in tokens:
            if word.lower() in sentiment_words:
                cleaned_tokens.append(word.lower())
            elif word.lower() not in stop_words_except_negations:
                cleaned_tokens.append(self.stemmer.stem(word.lower()))

        return ' '.join(cleaned_tokens)

    def load_data(self, file_path, text_column, label_column):
        df = pd.read_csv(file_path)

        print("Preprocessing text data...")
        df['processed_text'] = df[text_column].apply(self.preprocess_text)

        return df

    def prepare_features(self, df, text_column='processed_text', method='tfidf'):
        """Enhanced feature extraction for sentiment analysis"""
        if method == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=10000,  # Increase features
                ngram_range=(1, 2),  # Include bigrams
                min_df=2  # Ignore very rare terms
            )
        else:  # tfidf
            self.vectorizer = TfidfVectorizer(
                max_features=10000,  # Increase features
                ngram_range=(1, 2),  # Include bigrams
                min_df=2,  # Ignore very rare terms
                max_df=0.9,  # Ignore very common terms
                sublinear_tf=True  # Apply sublinear tf scaling
            )

        print("Extracting features...")
        X = self.vectorizer.fit_transform(df[text_column])
        print(f"Feature shape: {X.shape}")

        return X

    def train_model(self, X, y, model_type='naive_bayes'):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, C=1.0)  # Added regularization parameter
        else:  # naive_bayes
            self.model = MultinomialNB()

        # Train model
        print("Training model...")
        print(f"Training data shape: {X_train.shape}")
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)

        return X_train, X_test, y_train, y_test

    def predict_sentiment(self, text):
        """Predict sentiment for new text"""
        # Preprocess the input text
        processed_text = self.preprocess_text(text)

        # Transform to feature vector
        text_features = self.vectorizer.transform([processed_text])

        # Predict
        sentiment = self.model.predict(text_features)[0]

        # Get probability scores
        probabilities = self.model.predict_proba(text_features)[0]
        confidence = max(probabilities)

        return sentiment, confidence

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def visualize_important_features(self, n=20):
        """Visualize most important words for sentiment prediction"""
        if isinstance(self.model, LogisticRegression):
            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()

            # Get coefficients
            coefficients = self.model.coef_[0]

            # Create DataFrame for visualization
            top_features = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients
            })

            # Sort by absolute coefficient value
            top_features = top_features.reindex(top_features['coefficient'].abs().sort_values(ascending=False).index)

            # Plot top positive and negative features
            plt.figure(figsize=(12, 8))

            # Top positive features
            plt.subplot(1, 2, 1)
            sns.barplot(y=top_features['feature'][:n], x=top_features['coefficient'][:n], palette='Blues_d')
            plt.title(f'Top {n} Positive Features')
            plt.xlabel('Coefficient')

            # Top negative features
            plt.subplot(1, 2, 2)
            sns.barplot(y=top_features['feature'][-n:], x=top_features['coefficient'][-n:], palette='Reds_d')
            plt.title(f'Top {n} Negative Features')
            plt.xlabel('Coefficient')

            plt.tight_layout()
            plt.show()
        else:
            print("Feature importance visualization is only available for Logistic Regression model")
