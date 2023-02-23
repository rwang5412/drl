"""
Utility functions to change geoms online. This assumes A number of geoms are already initialized in
model XML file. This only supports box-only geom type.

Use cases
- Boxes for manipulation (need to be actual bodies so to another card)
- Stepping stones for discrete terrains
- Obstacles for navigation
- Stairs

Basic function
- Get/Set geom pose relative to any object
- Get/Set geom color
- Get/Set geom size

Copy over footstep generation code to vis geom on reset
move everything into this class to abstract
test dynamically add/remove during run time
think about useful fnunctions to caal in modular way

user will reset geoms into a thing at sim reset
- load reset for all geoms at once
- adjust robot to geoms
- reset simulation with a final call on mj_forward()

user will also change geoms on the fly
- directly tap in .sim to change geom properties.
- or call another class Box() that provides useful utilities to alter sim properties.

"""

import numpy as np
from env.util.quaternion import euler2quat

class Geom:
    def __init__(self, sim):
        self.sim = sim
        self.geoms = self.sim.box_geoms

        self.rise_noise  = [-0.02, 0.02]
        self.run_noise   = [-0.05, 0.05]
        self.roll_noise = [-0.00, 0.0]
        self.pitch_noise = [-0.15, 0.15]
        self.stair_width = 10

    def _create_step(self, box, start_x, start_y, start_z, length, rise, randomize_slope=False, slat=False):
        stair_rise   = rise + np.random.uniform(*self.rise_noise)
        stair_length = length + np.random.uniform(*self.run_noise)

        x = start_x + stair_length/2
        z = start_z

        vert_size = 0.01 if slat else np.abs(stair_rise)/2
        if np.sign(rise) < 0:
            vert_pos  = z + rise if slat else z + stair_rise/2
        else:
            vert_pos  = z if slat else z + stair_rise/2

        self.sim.set_geom_size(box, [stair_length/2, self.stair_width, vert_size])
        self.sim.set_geom_pose(box, [0, 0, 1, 1, 0, 0, 0])

        if randomize_slope:
            geom_plane = [np.random.uniform(*self.roll_noise), np.random.uniform(*self.pitch_noise), 0]
            quat_plane   = euler2quat(z=geom_plane[2], y=geom_plane[1], x=geom_plane[0])
            self.sim.set_geom_pose(box, [x, start_y, vert_pos, *quat_plane])

        return x + stair_length/2, z + stair_rise

    def create_stairs(self, start_x, start_y, start_z, slats=False, quat=False):
        default_length = np.random.uniform(0.24, 0.37)
        default_rise = np.random.uniform(0.1, 0.17)
        height    = np.random.choice(a=[1, 2, 3, 4, 5, 6, 7, 8])
        height = height-1 if height % 2 != 0 else height
        max_boxes = len(self.geoms)

        boxes = int(min(height*2, max_boxes-1))

        current_x = start_x + np.random.uniform(-1, 1)
        current_y = start_y
        current_z = start_z
        midway = boxes // 2

        for box in self.geoms[:midway]:
            current_x, current_z = self._create_step(box, current_x, current_y, current_z, default_length, default_rise, slat=slats, randomize_slope=quat)

        z_before = current_z
        intermediate = np.random.uniform(0.5, 2)
        current_x, current_z = self._create_step(self.geoms[midway], current_x, current_y, current_z, intermediate, default_rise, slat=slats, randomize_slope=quat)
        self.exempt_stairs = [midway]
        current_z = z_before

        for box in self.geoms[midway+1:boxes+1]:
            current_x, current_z = self._create_step(box, current_x, current_y, current_z, default_length, -default_rise, slat=slats, randomize_slope=quat)

        self.bounds = [start_x - default_length/2, current_x + default_length/2]

        for box in self.geoms[boxes+1:]:
            current_x += default_length/2
            self.sim.set_geom_size(box, [0.1, 0.1, 0.1])
            self.sim.set_geom_pose(box, [current_x, 20, 0, 1, 0, 0, 0])

    def _create_stone(self, box, start_x, start_y, start_z, rise, size):
        stair_rise   = rise
        stair_length = size

        x = start_x
        z = start_z

        vert_size = np.abs(stair_rise)/2
        if np.sign(rise) < 0:
            vert_pos  = z + stair_rise/2
        else:
            vert_pos  = z + stair_rise/2

        self.sim.set_geom_size(box, [stair_length, size, vert_size])
        self.sim.set_geom_pose(box, [x, start_y, vert_pos, 1, 0, 0, 0])
                
        return x + stair_length/2, z + stair_rise        

    def create_discrete_terrain(self):
        stone_array = 2*np.random.rand(20, 3) - 1
        stone_array[:,0:2] *= 1.5
        stone_array[:,2] = 0
        for i, box in enumerate(self.geoms):
            self._create_stone(box, *stone_array[i], rise=1, size=0.1)

    def create_obstacle(self):
        pass

    def check_step(self, x, y, z):
        ret_z   = z
        box_num = None
        for i, box in enumerate(self.geoms):
              box_d, box_w, box_h = self.sim.get_geom_size(name=box)
              box_x, box_y, box_z = self.sim.get_geom_pose(name=box)[0:3]

              if x < box_x + box_d and x > box_x - box_d and \
                  y < box_y + box_w and y > box_y - box_w:

                  ret_z = box_z + box_h
                  box_num = i
        return box_num, ret_z