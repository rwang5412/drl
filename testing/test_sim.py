from sim.CassieSim import MjCassieSim

def test_mj_sim():
    mj_sim = MjCassieSim()
    mj_sim.viewer_init()
    for i in range(100):
        for _ in range(50):
            mj_sim.step()
        mj_sim.viewer_render()
    # foo = input()