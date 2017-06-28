# Document classifier

This classifier classifies input documents into info labels.
  - Classifies using Bernoulli NaiveBayes
  - Supports only binary classification for now
  - Accuracy score is 51.34 % !!!   

### How to run:
##### input params:
  - -h, --help       show this help message and exit
  - --docs      Document directory
  - --labels   Classification file (CSV)
  - --quick    Quick run for testing / debug (not on all data) (default: False)
  - --clean Cleans the docs from stopwords and junk (will run slower)
#### Command example:
python document_classifier.py --docs docs/ --labels classification.csv --quick --clean

# python depandencies:
These are the python depandencies that you need to have insall in order to run the classifier:
 - argh
 - pandas
 - numpy
 - sklearn
 - scipy
 - BeautifulSoup

you can install them using

    $ sudo pip install packadge_name
