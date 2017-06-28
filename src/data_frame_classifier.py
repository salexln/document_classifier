from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score


class DataFrameClassifier(object):
    def __init__(self, data_frame):
        self._data_frame = data_frame
        self._classifier = None

    def classify_data_frame(self):
        x = self._data_frame['text']
        y = self._data_frame['class']
        X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

        print len(X_train), len(X_test), len(y_train), len(y_test)
        self._y_test = y_test
        count_vectorizer = CountVectorizer()
        counts = count_vectorizer.fit_transform(X_train.values)

        # self._classifier = MultinomialNB()
        self._classifier = BernoulliNB()
        self._train(x=counts, y=y_train.values)

        test = count_vectorizer.transform(X_test)
        self._test(test=test)

    def _train(self, x, y):
        self._classifier.fit(x, y)

    def _test(self, test):
        self._predictions = self._classifier.predict(test)

    def print_results(self):
        print self._predictions
        print self._y_test.values

        print accuracy_score(self._predictions, self._y_test.values)
