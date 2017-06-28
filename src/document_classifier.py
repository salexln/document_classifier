import argh
from parser import DocsParser


"""
Depandencies:
    argh
    pandas
    numpy
    sklearn
    scipy
    BeautifulSoup
"""


@argh.arg('--docs', help='Document directory', type=str, required=True)
@argh.arg('--labels', help='Classification file (CSV)', type=str, required=True)
@argh.arg('--quick', help='Quick run for testing / debug (not on all data)', default=False)
@argh.arg('--clean', help='Cleans the docs from stopwords and junk (will run slower)', default=False)
def main(**kwargs):
    parser = DocsParser(docs=kwargs['docs'],
                        labels=kwargs['labels'],
                        quick=kwargs['quick'],
                        clean=kwargs['clean'])
    parser.prepare_data()
    parser.classify()
    parser.print_results()


if __name__ == '__main__':
    parser = argh.ArghParser()
    parser.set_default_command(main)
    parser.dispatch()
