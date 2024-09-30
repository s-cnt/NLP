import os
import tarfile
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import pandas as pd
import time

# Download and extract dataset
def download_and_extract(url, file_name, data_dir):
    """Download and extract dataset if not already done."""
    if not os.path.exists(file_name):
        r = requests.get(url)
        with open(file_name, 'wb') as f:
            f.write(r.content)

    if not os.path.exists(data_dir):
        with tarfile.open(file_name, 'r:gz') as tar:
            tar.extractall()

# Read data from file
def read_data(file_path):
    """Reads and returns sentences from a file."""
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        return [line.strip() for line in file.readlines()]

# Clean and preprocess the text
def clean_and_preprocess(sentences):
    """Convert to lowercase, remove punctuation, stopwords, and apply stemming."""
    sentences = [sentence.lower() for sentence in sentences]
    sentences = [re.sub(r'[^\w\s]', '', sentence) for sentence in sentences]
    
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    stemmer = PorterStemmer()
    cleaned_sentences = [' '.join([stemmer.stem(word) for word in sentence.split() if word not in stop_words]) for sentence in sentences]
    
    return cleaned_sentences

# Split data into train, validation, and test sets
def split_data(positive_sentences, negative_sentences):
    """Split the dataset into train, validation, and test sets."""
    pos_train, pos_temp = train_test_split(positive_sentences, train_size=4000, shuffle=False)
    pos_val, pos_test = train_test_split(pos_temp, train_size=500, shuffle=False)
    neg_train, neg_temp = train_test_split(negative_sentences, train_size=4000, shuffle=False)
    neg_val, neg_test = train_test_split(neg_temp, train_size=500, shuffle=False)
    
    train_texts = pos_train + neg_train
    train_labels = [1] * 4000 + [0] * 4000
    val_texts = pos_val + neg_val
    val_labels = [1] * 500 + [0] * 500
    test_texts = pos_test + neg_test
    test_labels = [1] * 831 + [0] * 831
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

# Vectorize the text data using TF-IDF
def vectorize_text(train_texts, val_texts, test_texts):
    """Convert text data to TF-IDF vectors."""
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    
    return X_train, X_val, X_test

# Train models and evaluate their performance
def train_and_evaluate_models(X_train, train_labels, X_val, val_labels):
    """Train different models and evaluate them on the validation set."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, train_labels)
        val_predictions = model.predict(X_val)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_predictions, average='binary')

        results.append({
            "Model": name,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })
    
    results_df = pd.DataFrame(results)
    return results_df, models

# Evaluate the best model on the test set
def evaluate_best_model(models, best_model_name, X_test, test_labels):
    """Evaluate the best model on the test set and return the classification report and confusion matrix."""
    best_model = models[best_model_name]
    test_predictions = best_model.predict(X_test)
    
    report = classification_report(test_labels, test_predictions, output_dict=True)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_predictions).ravel()
    
    return pd.DataFrame(report).T, tn, fp, fn, tp

# Main function
def main():
    # 1. Download and extract the dataset
    url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    file_name = 'rt-polaritydata.tar.gz'
    data_dir = './rt-polaritydata/'
    
    download_and_extract(url, file_name, data_dir)

    # 2. Read and preprocess the data
    pos_file_path = os.path.join(data_dir, 'rt-polarity.pos')
    neg_file_path = os.path.join(data_dir, 'rt-polarity.neg')
    
    positive_sentences = read_data(pos_file_path)
    negative_sentences = read_data(neg_file_path)
    
    positive_sentences = clean_and_preprocess(positive_sentences)
    negative_sentences = clean_and_preprocess(negative_sentences)

    # 3. Split the data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_data(positive_sentences, negative_sentences)

    # 4. Vectorize the text
    X_train, X_val, X_test = vectorize_text(train_texts, val_texts, test_texts)

    # 5. Train models and evaluate
    print("Training models...")
    time.sleep(1)
    results_df, models = train_and_evaluate_models(X_train, train_labels, X_val, val_labels)

    # 6. Display results
    print("\n### Model Performance ###")
    print(results_df)

    # 7. Evaluate best model on the test set
    best_model_name = results_df.loc[results_df['F1 Score'].idxmax()]['Model']
    print(f"\n### Best Model: {best_model_name} ###")

    report, tn, fp, fn, tp = evaluate_best_model(models, best_model_name, X_test, test_labels)
    
    print("#### Test Set Report ####")
    print(report)

    print(f"\n#### Confusion Matrix for {best_model_name} ####")
    print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")

# Execute the main function
if __name__ == "__main__":
    main()
