import re
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
import gensim

class ClassicalMLPipeline:
    def __init__(self, vector_size=100, window=5, min_count=2, model_type="svm"):
        self.w2v_model = None
        self.clf = None
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model_type = model_type
        self.label_names = None

    def preprocess(self, text):
        """Simple text normalizer: lowercase, remove non-alphanumeric, split."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()

    def get_sentence_vector(self, tokens):
        """Mean of word vectors for a given list of tokens."""
        valid_words = [word for word in tokens if word in self.w2v_model.wv]
        if not valid_words:
            return np.zeros(self.vector_size)
        return np.mean([self.w2v_model.wv[word] for word in valid_words], axis=0)

    def fit(self, df_train, label_names):
        print("Preprocessing training texts...")
        self.label_names = label_names
        train_tokens = df_train['text'].apply(self.preprocess).tolist()
        
        print(f"Training Word2Vec model (size={self.vector_size})...")
        self.w2v_model = gensim.models.Word2Vec(
            sentences=train_tokens, 
            vector_size=self.vector_size, 
            window=self.window, 
            min_count=self.min_count, 
            workers=4
        )
        
        print("Generating training sentence embeddings...")
        X_train = np.array([self.get_sentence_vector(tokens) for tokens in train_tokens])
        y_train = df_train['label'].values
        
        if self.model_type == "svm":
            print("Training SVMClassifier...")
            self.clf = SVC(kernel='linear', C=1.0)
        else:
            print("Training RandomForest...")
            self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
            
        self.clf.fit(X_train, y_train)
        print("Classical ML training complete.")

    def predict(self, texts):
        """Predicts labels for a list of raw string texts."""
        tokens = [self.preprocess(text) for text in texts]
        X = np.array([self.get_sentence_vector(toks) for toks in tokens])
        preds = self.clf.predict(X)
        return [self.label_names[p] for p in preds]

    def evaluate(self, df_test):
        print("Evaluating Classical ML Pipeline on test set...")
        y_true = df_test['label'].values
        tokens = df_test['text'].apply(self.preprocess).tolist()
        X_test = np.array([self.get_sentence_vector(toks) for toks in tokens])
        
        y_pred = self.clf.predict(X_test)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"Classical ML Accuracy: {acc:.4f}")
        print(f"Classical ML F1 Score (weighted): {f1:.4f}")
        return acc, f1
