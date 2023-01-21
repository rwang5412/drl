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
