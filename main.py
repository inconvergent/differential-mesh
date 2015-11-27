#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from numpy import pi
from numpy import sqrt
from numpy import zeros
from numpy import cos
from numpy import sin
from numpy.random import random

from modules.timers import named_sub_timers

NMAX = 10e7
SIZE = 5000
ONE = 1./SIZE

RAD = 3*ONE
H = sqrt(3.)*RAD
NEARL = 2*RAD
FARL = RAD*20

OPT_STP = 1./SIZE*0.5

MID = 0.5

LINEWIDTH = 1*ONE

BACK = [1,1,1,1]
FRONT = [0,0,0,0.3]
RED = [1,0,0,0.05]
BLUE = [0,0,1,0.3]
GREEN = [0,1,0,0.3]

PROCS = 6


TWOPI = pi*2.

np_coord = zeros((NMAX,6), 'float')

i = 0


def show(render, dm):

  global np_coord
  global i

  render.clear_canvas()

  #num = dm.np_get_triangles_coordinates(np_coord)
  num = dm.np_get_triangles_coordinates(np_coord)
  render_random_triangle = render.random_triangle
  rgba = FRONT
  render.set_front(rgba)

  for e,vv in enumerate(np_coord[:num,:]):

    render_random_triangle(*vv,grains=80)

  render.write_to_png('res/res_a_{:05d}.png'.format(i))

  i += 1


def main():

  from differentialMesh import DifferentialMesh
  from render.render import Render
  from time import time
  from modules.helpers import print_stats
  from numpy import array


  DM = DifferentialMesh(NMAX, 2*FARL, NEARL, FARL, PROCS)

  DM.new_faces_in_ngon(MID, MID, H, 6, 0.0)

  render = Render(SIZE, BACK, FRONT)
  render.set_line_width(LINEWIDTH)

  st = named_sub_timers()

  tsum = 0

  minimum_length = H*0.8
  maximum_length = H*2

  for i in xrange(10000):

    t1 = time()

    st.start()
    DM.optimize_position(OPT_STP)
    st.t('opt')

    henum = DM.get_henum()
    st.t('rnd')

    surface_edges = array(
      [DM.is_surface_edge(i)>0 \
      for i in range(henum)],
      'bool').nonzero()[0]

    st.t('surf')

    rnd = random(len(surface_edges)*2)
    the = (1.-2*rnd[::2])*pi
    rad = rnd[1::2]*0.5*H
    dx = cos(the)*rad
    dy = sin(the)*rad
    st.t('rnd2')

    for i,se in enumerate(surface_edges):

      DM.new_triangle_from_surface_edge(
        se,
        H,
        dx[i],
        dy[i],
        minimum_length=minimum_length,
        maximum_length=maximum_length,
        merge_ragged_edge=1
      )
      st.t('tri')

    st.start()
    DM.optimize_edges(H*2, NEARL*0.5)
    st.t('opte')


    tsum += time() - t1

    if i%40==0:

      print_stats(render.num_img, tsum, DM)

      show(render, DM)

      tsum = 0

    st.p()



if __name__ == '__main__' :

    main()

