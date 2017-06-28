
from BeautifulSoup import BeautifulSoup
import codecs
import os
import csv
from pandas import DataFrame
from data_frame_classifier import DataFrameClassifier


class DocsParser(object):
    def __init__(self, docs, labels, quick, clean):
        self._docs_path = docs
        self._labels_path = labels
        self._quick = quick
        self._clean = clean
        self._data = []
        self._df = None
        self._test = None
        self._predictions = None
        self._labels = []
        self._y_test = None
        self._classifier = None

    def _html_to_text(self, path):
        with codecs.open(path, 'r', 'utf-8') as html_file:
            document = BeautifulSoup(html_file.read()).getText()
            return document

    def _get_data_files(self):
        files_path = []
        files = os.listdir(self._docs_path)

        for idx, file in enumerate(files):
            files_path.append(os.path.join(self._docs_path, file))

            if idx > 19 and self._quick:
                break
        return files_path

    def _get_labels(self):
        with open(self._labels_path, 'r') as file:
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

                self._labels.append((doc_id, doc_class))

                if idx > 20 and self._quick:
                    break

    def _build_data_frame(self):
        rows = []

        for idx, doc in enumerate(self._data):
            rows.append({'text': doc, 'class': self._labels[idx][1]})

        data_frame = DataFrame(rows)
        self._df = data_frame

    def prepare_data(self):
            files = self._get_data_files()

            for file in files:                
                text = self._html_to_text(file)
                self._clean_text(text=text)
                self._data.append(text)

            print 'number of files:', len(files)
            print 'data len:', len(self._data)

            self._get_labels()
            print 'Number of labels:', len(self._labels)

            self._build_data_frame()

    def _clean_text(self, text):
        if self._clean:
            temp_text = ''
            for word in text.split(' '):
                word = word.strip()
                temp_text += word
                temp_text += ' '
            text = temp_text

    def classify(self, model):
        self._classifier = DataFrameClassifier(data_frame=self._df, model=model)
        self._classifier.classify_data_frame()

    def print_results(self):
        self._classifier.print_results()
