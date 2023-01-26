import argparse
import asyncio
import os
import pickle
import sys

import testing.test_sim as ts
import testing.test_env as test_env

from testing.test_ar_sim import (
    test_ar_connect,
    test_ar_api_goto,
    test_ar_sim_forward,
    test_ar_sim_llapi_walking_handover,
    test_ar_sim_llapi_teststand
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", default=False, action='store_true')
    parser.add_argument("--env", default=False, action='store_true')
    parser.add_argument("--sim", default=False, action='store_true')
    parser.add_argument("--ar", default=False, action='store_true')
    args = parser.parse_args()

    if args.algo:
        pass
    if args.env:
        test_env.test()
    if args.sim:
        ts.test_all_sim()
    else:
        ts.test_mj_sim()
    if args.ar:
        asyncio.run(test_ar_connect())
        asyncio.run(test_ar_api_goto())
        asyncio.run(test_ar_sim_forward())
        asyncio.run(test_ar_sim_llapi_teststand())
        asyncio.run(test_ar_sim_llapi_walking_handover())

