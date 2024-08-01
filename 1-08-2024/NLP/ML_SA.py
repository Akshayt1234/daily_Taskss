import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Create a sample dataset with labels
data = [
    ("I love NLP", "Positive"),
    ("I hate this technology", "Negative"),
    ("It's okay, nothing special", "Neutral")
]

# Separate sentences and labels
sentences, labels = zip(*data)

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the stopwords
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    # Remove stop words and punctuations
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

# Preprocess sentences
preprocessed_sentences = [preprocess(sentence) for sentence in sentences]

# Feature extraction
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(preprocessed_sentences)  # Use fit_transform here

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(x_train, y_train)

# Predict on the test set
y_pred = classifier.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
