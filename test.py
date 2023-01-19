import sys
import os
import pickle
import argparse
import testing.test_sim as ts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", default=False, action='store_true')
    parser.add_argument("--env", default=False, action='store_true')
    parser.add_argument("--sim", default=False, action='store_true')
    args = parser.parse_args()

    if args.algo:
        pass
    if args.env:
        pass
    if args.sim:
        ts.test_all_sim()
    else:
        ts.test_mj_sim()
