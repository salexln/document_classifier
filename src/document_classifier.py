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
@argh.arg('--quick', help='Quick run for testing (not on all data)', default=False)
def main(**kwargs):
    parser = DocsParser(docs=kwargs['docs'],
                        labels=kwargs['labels'],
                        quick=kwargs['quick'])
    parser.prepare_data()
    parser.classify()
    parser.print_results()


if __name__ == '__main__':
    parser = argh.ArghParser()
    parser.set_default_command(main)
    parser.dispatch()
