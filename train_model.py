from sentiment_analyzer import SentimentAnalyzer
import pandas as pd
import pickle


def train_and_save_model():
    analyzer = SentimentAnalyzer()

    # Load data
    data = analyzer.load_data('data/IMDB Dataset.csv', 'review', 'sentiment')

    # Print data info for debugging
    print(f"Data shape: {data.shape}")
    print(f"Label distribution: \n{data['sentiment'].value_counts()}")

    # Convert labels if needed
    # If your labels are "positive"/"negative" strings, convert to 1/0
    if data['sentiment'].dtype == object:
        print("Converting string labels to binary...")
        data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

    X = analyzer.prepare_features(data, method='tfidf')
    y = data['sentiment']

    # Check for NaN values
    print(f"Any NaN in features: {X.sum() == 0}")

    X_train, X_test, y_train, y_test = analyzer.train_model(X, y, model_type='logistic')

    analyzer.visualize_important_features()

    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump(analyzer, f)

    print("Model trained and saved successfully!")


if __name__ == "__main__":
    train_and_save_model()