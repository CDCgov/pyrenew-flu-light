"""
File for analyzing results of experimental
runs, including plotting & model comparison.
"""

import argparse


def main(args):
    pass


if __name__ == "__main__":
    # use argparse for command line running
    parser = argparse.ArgumentParser(
        description="Analyze experiment results from pyrenew-flu-light."
    )
    parser.add_argument(
        "--reporting_date",
        type=str,
        required=True,
        help="The reporting date.",
    )
    parser.add_argument(
        "--historical_data",
        action="store_true",
        help="Load model weights before training.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="The name of a given experiment.",
    )
    args = parser.parse_args()
    main(args)
