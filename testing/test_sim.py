from sim.CassieSim import MjCassieSim, LibCassieSim

def test_mj_sim():
    mj_sim = MjDigitSim()
    mj_sim.viewer_init()
    while mj_sim.viewer.is_alive:
        if not mj_sim.viewer.paused:
            for _ in range(50):
                mj_sim.step()
        mj_sim.viewer_render()

def test_all_sim():
    # TODO: Add other sims to this list after implemented
    sim_list = [LibCassieSim]#, MjCassieSim]
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'
    for sim in sim_list:
        print(f"Testing {sim.__name__}")
        test_sim_init(sim)
        test_sim_step(sim)
        test_sim_viewer(sim)
        print(f"{OKGREEN}{sim.__name__} passed all tests.{ENDC}")

def test_sim_init(sim):
    print("Making sim")
    test_sim = sim()
    print("Passed made sim")

def test_sim_step(sim):
    print("Testing sim step")
    test_sim = sim()
    for i in range(100):
        test_sim.step()
    print("Passed sim step")

def test_sim_viewer(sim):
    print("Testing sim viewer, quit window to continue")
    test_sim = sim()
    test_sim.viewer_init()
    while test_sim.viewer.is_alive:
        if not test_sim.viewer.paused:
            for _ in range(50):
                test_sim.step()
        test_sim.viewer_render()
    print("Passed sim viewer")
