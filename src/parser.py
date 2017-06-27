
from BeautifulSoup import BeautifulSoup
import codecs
import os
import csv
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score



def html_to_text(path):
    with codecs.open(path, 'r', 'utf-8') as html_file:
        document = BeautifulSoup(html_file.read()).getText()
        return document

# text = html_to_text('../data/docs/9.html')


def get_all_files(path):
    files_path = []
    files = os.listdir(path)

    for idx, file in enumerate(files):
        files_path.append(os.path.join(path, file))
        # if idx > 19:
        #     break
    return files_path


def get_files_classification(path):
    classes = []
    with open(path, 'r') as file:
        lines = file.readlines()

        header = True
        for idx, line in enumerate(lines):
            if header:
                header = False
                continue

            doc_id = int(line.split(',')[0])
            doc_class = int(line.split(',')[1])
            if doc_class != 1:
                doc_class = 0

            classes.append((doc_id, doc_class))

            # if idx > 20:
            #     break

        return classes


def build_data_frame(docs, classification):
    rows = []

    for idx, doc in enumerate(docs):
        rows.append({'text': doc, 'class': classification[idx][1]})

    data_frame = DataFrame(rows)
    return data_frame


def classify_data_frame(df):
    x = df['text']
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    print len(X_train), len(X_test), len(y_train), len(y_test)

    count_vectorizer = CountVectorizer()
    counts = count_vectorizer.fit_transform(X_train.values)

    # train:
    # classifier = MultinomialNB()
    classifier = BernoulliNB()
    targets = y_train.values
    classifier.fit(counts, targets)

    # test:
    test = count_vectorizer.transform(X_test)
    predictions = classifier.predict(test)

    print predictions
    print y_test.values

    print accuracy_score(predictions, y_test.values)


if __name__ == '__main__':
    data_path = '../data/docs'
    files = get_all_files(path=data_path)


    data = []
    for file in files:
        text = html_to_text(file)
        data.append(text)

    print 'number of files:', len(files)
    print 'data len:', len(data)

    res_path = '../data/File_Classification.csv'
    classes = get_files_classification(path=res_path)

    print 'number of classes:', len(classes)

    df = build_data_frame(docs=data, classification=classes)

    classify_data_frame(df=df)
