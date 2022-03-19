"""
Author: Wenyu Ouyang
Date: 2022-01-25 16:49:00
LastEditTime: 2022-03-19 16:39:52
LastEditors: Wenyu Ouyang
Description: to test if the arg-parse commands are correct
FilePath: /HydroBench/hydrodataset/app/script_args.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import argparse
import sys

from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))


def main(args):
    """
    Main function which is called from the command line. Entrypoint for training all ML models.
    """
    print("the years: ", list(range(int(args.year_range[0]), int(args.year_range[1]))))


# python script_args.py --year_range 1990 2011
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Daymet within the boundary of each basin in CAMELS"
    )
    parser.add_argument(
        "--year_range",
        dest="year_range",
        help="The start and end years (right open interval)",
        default=[1980, 2021],
        nargs="+",
    )
    the_args = parser.parse_args()
    main(the_args)
