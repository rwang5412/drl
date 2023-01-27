import argparse
import asyncio
import os
import pickle
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", default=False, action='store_true')
    parser.add_argument("--env", default=False, action='store_true')
    parser.add_argument("--sim", default=False, action='store_true')
    parser.add_argument("--ar", default=False, action='store_true')
    parser.add_argument("--render", default=False, action='store_true')
    args = parser.parse_args()

    if args.algo:
        pass
    if args.env:
        pass
    if args.sim:
        import testing.test_sim as ts
        ts.test_all_sim()
    if args.render:
        from testing.test_offscreen_render import test_offscreen_rendering
        test_offscreen_rendering()
    if args.ar:        
        from testing.test_ar_sim import (
            test_ar_connect,
            test_ar_api_goto,
            test_ar_sim_forward,
            test_ar_sim_llapi_walking_handover,
            test_ar_sim_llapi_teststand
        )
        asyncio.run(test_ar_connect())
        asyncio.run(test_ar_api_goto())
        asyncio.run(test_ar_sim_forward())
        asyncio.run(test_ar_sim_llapi_teststand())
        asyncio.run(test_ar_sim_llapi_walking_handover())

