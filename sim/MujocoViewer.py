import mujoco as mj
import glfw
import numpy as np
from threading import Lock

"""
Class to handle visualization of generic Mujoco models. Adapted from cassie-mujoco-sim
(https://github.com/osudrl/cassie-mujoco-sim) and mujoco-python-viewer
(https://github.com/rohanpsingh/mujoco-python-viewer)
"""

# help strings
# TODO: Fill out rest of help strings for key functionality
HELP_CONTENT = ("Alt mouse button\n"
        "UI right hold\n"
        "UI title double-click\n"
        "Space\n"
        "Esc\n"
        "Right arrow\n"
        "Left arrow\n"
        "Down arrow\n"
        "Up arrow\n"
        "Page Up\n"
        "Double-click\n"
        "Right double-click\n"
        "Ctrl Right double-click\n"
        "Scroll, middle drag\n"
        "Left drag\n"
        "[Shift] right drag\n"
        "Ctrl [Shift] drag\n"
        "Ctrl [Shift] right drag")

HELP_TITLE = ("Swap left-right\n"
        "Show UI shortcuts\n"
        "Expand/collapse all  \n"
        "Pause\n"
        "Free camera\n"
        "Step forward\n"
        "Step back\n"
        "Step forward 100\n"
        "Step back 100\n"
        "Select parent\n"
        "Select\n"
        "Center\n"
        "Track camera\n"
        "Zoom\n"
        "View rotate\n"
        "View translate\n"
        "Object rotate\n"
        "Object translate")

class MujocoViewer():
    def __init__(self,
                 model,
                 data,
                 reset_qpos,
                 width=None,
                 height=None):

        self.model = model
        self.data = data
        self.reset_qpos = reset_qpos
        assert len(self.reset_qpos) == self.model.nq, \
               f"Size of MujocoViewer reset qpos does not match model nq size"

        self.is_alive = True
        self.paused = True

        glfw.init()
        if not width:
            width, _ = glfw.get_video_mode(glfw.get_primary_monitor()).size

        if not height:
            _, height = glfw.get_video_mode(glfw.get_primary_monitor()).size
        self.window = glfw.create_window(
            width, height, "", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(
            self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = framebuffer_width * 1.0 / window_width
        self.viewport = mj.MjrRect(
            0, 0, framebuffer_width, framebuffer_height)

        # set callbacks
        glfw.set_cursor_pos_callback(
            self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(
            self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)

        # create options, camera, scene, context
        self.vopt = mj.MjvOption()
        self.cam = mj.MjvCamera()
        self.scn = mj.MjvScene(self.model, maxgeom=1000)
        self.pert = mj.MjvPerturb()
        self.ctx = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        self.fontscale = 150
        # Set up mujoco visualization objects
        self.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = 0
        self.cam.fixedcamid = -1
        self.cam.distance = 3
        self.cam.azimuth = 90
        self.cam.elevation = -20
        self.vopt.flags[11] = 1    # Render applied forces

        # Set interaction ctrl vars
        self.paused = True
        self._lastbutton = glfw.MOUSE_BUTTON_1
        self._lastclicktm = 0.0
        self._showhelp = False
        self._showoption = False
        self._showGRF = False
        self._GRFcount = 0
        self._showfullscreen = False
        self._showsensor = False
        self._slowmotion = False
        self._showinfo = True
        self._framenum = 0
        self._lastframenum = 0
        self._marker_num = 0
        self._perturb_body = 1
        self._pipe_video_out = None
        self._perturb_force = np.zeros(6)

        self._gui_lock = Lock()
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._last_left_click_time = None
        self._last_right_click_time = None
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._image_idx = 0
        self._image_path = "/tmp/frame_%07d.png"
        self._time_per_render = 1 / 60.0
        self._run_speed = 1.0
        self._loop_count = 0
        self._advance_by_one_step = False
        self._hide_menus = False

    def render(self):
        if glfw.window_should_close(self.window):
            glfw.destroy_window(self.window)
            self.window = None
            self.is_alive = False
            return
        # # Apply perturbations
        if self.pert.select > 0:
            self.data.xfrc_applied = np.zeros((self.model.nbody, 6))
            mj.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)    # move mocap bodies only
            mj.mjv_applyPerturbForce(self.model, self.data, self.pert)
        with self._gui_lock:
            mj.mjv_updateScene(
                self.model,
                self.data,
                self.vopt,
                self.pert,
                self.cam,
                mj.mjtCatBit.mjCAT_ALL.value,
                self.scn)
            mj.mjr_render(self.viewport, self.scn, self.ctx)
            if self._showhelp:
                mj.mjr_overlay(mj.mjtFontScale.mjFONTSCALE_150.value,
                               mj.mjtGridPos.mjGRID_TOPLEFT,
                               self.viewport,
                               HELP_TITLE,
                               HELP_CONTENT,
                               self.ctx)
            if self._showinfo:
                if self.paused:
                    str_paused = "\nPaused"
                else:
                    str_paused = "\nRunning"
                str_paused += "\nTime:"
                if self._slowmotion:
                    str_slow = "(10x slowdown)"
                else:
                    str_slow = ""
                str_info = str_slow + f"\n\n{self.data.time:.2f}"
                mj.mjr_overlay(mj.mjtFontScale.mjFONTSCALE_150.value,
                               mj.mjtGridPos.mjGRID_BOTTOMLEFT,
                               self.viewport,
                               str_paused,
                               str_info,
                               self.ctx)
            glfw.swap_buffers(self.window)
        glfw.poll_events()

    def _key_callback(self, window, key, scancode, action, mods):
        if action == glfw.RELEASE:
            # Don't do anything if released the key
            return
        elif action == glfw.PRESS:
            # If press 'P' with no mods, then attach camera to center of model
            if key == glfw.KEY_P and mods == 0:
                self.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
                self.cam.trackbodyid = 0
                self.cam.fixedcamid = -1
                self.cam.distance = 3
                self.cam.azimuth = 90
                self.cam.elevation = -20
            # Handle control key mod
            if mods == glfw.MOD_CONTROL:
                if key == glfw.KEY_A:
                    self.cam.lookat = self.model.stat.center
                    self.cam.distance = 1.5 * self.model.stat.extent
                    self.cam.type = mj.mjtCamera.mjCAMERA_FREE
                elif key == glfw.KEY_P: # Print out qpos
                    qpos_str = "qpos: "
                    for i in range(self.model.nq):
                        qpos_str += f"{self.d.qpos[i]:.4f}"
                        if i != self.model.nq - 1:
                            qpos_str += ", "
                    print(qpos_str)
                elif key == glfw.KEY_T: # Save screenshot
                    img = np.zeros(
                        (glfw.get_framebuffer_size(
                            self.window)[1], glfw.get_framebuffer_size(
                            self.window)[0], 3), dtype=np.uint8)
                    mj.mjr_readPixels(img, None, self.viewport, self.ctx)
                    imageio.imwrite(self._image_path % self._image_idx, np.flipud(img))
                    self._image_idx += 1
                elif key == glfw.KEY_Q:
                    glfw.set_window_should_close(self.window, True)
            # Toggle visualization flags
            for i in range(mj.mjtVisFlag.mjNVISFLAG):
                if key == mj.mjVISSTRING[i][2][0]:
                    self.opt.flags[i] = 1 - self.opt.flags[i]
                    # return
                    # Don't return here due to overlapping key in VISSTRING and RNDSTRING. "," key
                    # is in both, so if "," is pressed toggle both flags and return in RNDSTRING check
            # Toggle rendering flags
            for i in range(mj.mjtRndFlag.mjNRNDFLAG):
                if key == mj.mjRNDSTRING[i][2]:
                    self.scn.flags[i] = 1 - self.scn.flags[i]
                    return
            # Toggle geom/site group
            for i in range(mj.mjNGROUP):
                if key == i + 48:   # Int('0') = 48
                    if mods and glfw.MOD_SHIFT == True:
                        # NOTE: what is self.opt? this seems broken. 
                        self.vopt.sitegroup[i] = 1 - self.vopt.sitegroup[i]
                    else:
                        self.vopt.geomgroup[i] = 1 - self.vopt.geomgroup[i]
        # Handle regular inidividual key presses
        if key == glfw.KEY_F1:              # help
            self._showhelp = not self._showhelp
        elif key == glfw.KEY_F2:            # option
            self._showoption = not self._showoption
        elif key == glfw.KEY_F3:            # info
            self._showinfo = not self._showinfo
        elif key == glfw.KEY_F4:            # GRF
            self._showGRF = not self._showGRF
        elif key == glfw.KEY_F5:            # sensor figure
            self._showsensor = not self._showsensor
        elif key == glfw.KEY_F6:            # toggle fullscreen
            self._showfullscreen = not self._showfullscreen
        elif key == glfw.KEY_ENTER:         # toggle slow motion
            self._slowmotion = not self._slowmotion
        elif key == glfw.KEY_SPACE:         # pause
            self.paused = not self.paused
        elif key == glfw.KEY_BACKSPACE:     # reset
            mj.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self.reset_qpos
            mj.mj_forward(self.model, self.data)
        elif key == glfw.KEY_RIGHT:         # step forward
            if self.paused:
                mj.mj_step(self.model, self.data)
        elif key == glfw.KEY_DOWN:          # step forward 100
            if self.paused:
                mj.mj_step(self.model, self.data, nstep=100)
        elif key == glfw.KEY_ESCAPE:        # free camera
            self.cam.type = mj.mjtCamera.mjCAMERA_FREE
        elif key == glfw.KEY_EQUAL:         # bigger font
            if self.fontscale < 200:
                fontscale += 50
                self.ctx = mj.MjrContext(self.model, fontscale)
        elif key == glfw.KEY_MINUS:         # smaller font
            if self.fonstacle > 100:
                fontscale -= 50
                self.ctx = mj.MjrContext(self.model, fontscale)
        elif key == glfw.KEY_LEFT_BRACKET:  # previous fixed camera or free
            if self.model.ncam > 0 and self.cam.type == mj.mjtCamera.mjCAMERA_FIXED:
                if self.cam.fixedcamid > 0:
                    self.cam.fixedcamid -= 1
                else:
                    self.cam.type = mj.mjtCamera.mjCAMERA_FREE
        elif key == glfw.KEY_RIGHT_BRACKET:  # next fixed camera
            if self.model.ncam > 0:
                if self.cam.type != mj.mjtCamera.mjCAMERA_FIXED:
                    self.cam.type = mj.mjtCamera.mjCAMERA_FIXED
                elif self.cam.fixedcamid < self.model.ncam - 1:
                    self.cam.fixedcamid += 1

        return

    def _cursor_pos_callback(self, window, xpos, ypos):
        # no buttons down, nothing to do
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        # compute mouse displacement
        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)
        width, height = glfw.get_framebuffer_size(window)

        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
            glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

        # determine action based on mouse button
        if self._button_right_pressed:
            action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left_pressed:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        # move perturb or camera
        with self._gui_lock:
            if self.pert.active:
                mj.mjv_movePerturb(
                    self.model,
                    self.data,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.pert)
            else:
                mj.mjv_moveCamera(
                    self.model,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.cam)

    def _mouse_button_callback(self, window, button, act, mods):
        # update button state
        self._button_left_pressed = button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS
        self._button_right_pressed = button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS

        # Alt: swap left and right
        if mods == glfw.MOD_ALT:
            tmp = self._button_left_pressed
            self._button_left_pressed = self._button_right_pressed
            self._button_right_pressed = tmp
            if button == glfw.MOUSE_BUTTON_LEFT:
                button = glfw.MOUSE_BUTTON_RIGHT
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                button = glfw.MOUSE_BUTTON_LEFT

        # update mouse position
        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

        # set perturbation
        newperturb = 0
        if mods == glfw.MOD_CONTROL and self.pert.select > 0:
            if act == glfw.PRESS:
                # Disable vis perturb force when using mouse perturb, only want to vis perturb object
                self.vopt.flags[11] = 0
                # right: translate, left: rotate
                if self._button_right_pressed:
                    newperturb = mj.mjtPertBit.mjPERT_TRANSLATE
                elif self._button_left_pressed:
                    newperturb = mj.mjtPertBit.mjPERT_ROTATE

                # perturbation onste: reset reference
                if newperturb and not self.pert.active:
                    mj.mjv_initPerturb(self.model, self.data, self.scn, self.pert)
            else:
                # Enable vis perturn force again
                self.vopt.flags[11] = 1
        self.pert.active = newperturb

        # detect a left- or right- doubleclick (250 msec)
        curr_time = glfw.get_time()
        if act == glfw.PRESS and (curr_time - self._lastclicktm < 0.25) and (button == self._lastbutton):
            # determine selection mode
            selmode = 2         # Right Click
            if button == glfw.MOUSE_BUTTON_LEFT:
                selmode = 1
            elif mods == glfw.MOD_CONTROL:
                selmode = 3     # CTRL + Right Click

            # find geom and 3D click point, get corresponding body
            width, height = self.viewport.width, self.viewport.height
            aspectratio = width / height
            relx = x / width
            rely = (self.viewport.height - y) / height
            selpnt = np.zeros((3, 1), dtype=np.float64)
            selgeom = np.zeros((1, 1), dtype=np.int32)
            selskin = np.zeros((1, 1), dtype=np.int32)
            selbody = mj.mjv_select(self.model, self.data, self.vopt, aspectratio, relx, rely,
                self.scn, selpnt, selgeom, selskin)

            # set lookat point, start tracking is requested
            if selmode == 2 or selmode == 3:
                # set cam lookat
                if selbody >= 0:
                    self.cam.lookat = selpnt.flatten()
                # switch to tracking camera if dynamic body clicked
                if selmode == 3 and selbody > 0:
                    self.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
                    self.cam.trackbodyid = selbody
                    self.cam.fixedcamid = -1
            # set body selection
            else:
                if selbody >= 0:
                    # record selection
                    self.pert.select = selbody
                    self.pert.skinselect = selskin
                    # compute localpos
                    vec = selpnt.flatten() - self.data.xpos[selbody]
                    mat = self.data.xmat[selbody].reshape(3, 3)
                    self.pert.localpos = self.data.xmat[selbody].reshape(
                        3, 3).dot(vec)
                else:
                    self.pert.select = 0
                    self.pert.skinselect = -1
            # stop perturbation on select
            self.pert.active = 0

        if act == glfw.PRESS:
            self._lastbutton = button
            self._lastclicktm = glfw.get_time()
        else:   # 3D release
            self.pert.active = 0

    def _scroll_callback(self, window, x_offset, y_offset):
        with self._gui_lock:
            mj.mjv_moveCamera(self.model,
                              mj.mjtMouse.mjMOUSE_ZOOM,
                              0,
                              -0.05 * y_offset,
                              self.scn,
                              self.cam)


