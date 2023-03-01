import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", default=False, action='store_true')
    parser.add_argument("--env", default=False, action='store_true')
    parser.add_argument("--sim", default=False, action='store_true')
    parser.add_argument("--ar", default=False, action='store_true')
    parser.add_argument("--clock", default=False, action='store_true')
    parser.add_argument("--nn", default=False, action='store_true')
    parser.add_argument("--render", default=False, action='store_true')
    parser.add_argument("--all", default=False, action='store_true')
    args = parser.parse_args()

    if args.all:
        args.algo = True
        args.env = True
        args.sim = True
        args.clock = True
        args.nn = True

    if args.algo:
        test_all_algos()
    if args.env:
        import testing.test_env as test_env
        test_env.test_all_env()
    if args.sim:
        import testing.test_sim as test_sim
        test_sim.test_all_sim()
    if args.render:
        from testing.test_offscreen_render import test_offscreen_rendering
        test_offscreen_rendering()
    if args.ar:        
        import asyncio
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
    if args.clock:
        from testing.test_clock import test_all_clocks
        test_all_clocks()
    if args.nn:
        from testing.test_nn import test_nn
        test_nn()
    if args.mirror:
        from testing.test_mirror import test_mirror
        test_mirror()
