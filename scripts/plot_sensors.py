from argparse import ArgumentParser

import matplotlib.pyplot as plt

from fnirsatlas.load_parser import load_parser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True)
    return parser.parse_args()


def main():
    parser = load_parser(parse_args().dataset, download="no")
    subject_id = parser.get_subject_ids()[0]
    raw = parser.convert_to_raw(subject_id)
    raw.plot_sensors(show_names=True)
    raw.get_montage().plot(show_names=True)
    plt.show()


if __name__ == "__main__":
    main()
