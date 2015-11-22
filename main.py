#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from numpy import pi
from numpy import sqrt
from numpy import zeros
from numpy import cos
from numpy import sin
from numpy import power
#from numpy import floor
from numpy.random import random

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


TWOPI = pi*2.

np_coord = zeros((NMAX,6), 'float')
np_edges_coord = zeros((NMAX,4), 'float')
np_gen = zeros(NMAX, 'int')

i = 0


def show(render, dm):

  global np_coord
  global i

  render.clear_canvas()

  render_random_uniform_circle = render.random_uniform_circle

  #num = dm.np_get_triangles_coordinates(np_coord)
  num = dm.np_get_edges_coordinates(np_edges_coord)
  render_random_triangle = render.random_triangle
  dm.np_get_edges_gen(np_gen)
  #render_triangle = render.triangle

  dd = sqrt(power(np_edges_coord[:,2]-np_edges_coord[:,0],2) +
       power(np_edges_coord[:,3]-np_edges_coord[:,1],2))*0.5

  for e,vv in enumerate(np_edges_coord[:num,:]):

    if np_gen[e]%2==0:
      rgba = RED
    else:
      rgba = FRONT

    render.set_front(rgba)

    #render.set_front(FRONT)
    #render_random_triangle(*vv,grains=80)
    render_random_uniform_circle(vv[0], vv[1], dd[e], grains=30, dst=0)
    #render_random_parallelogram(*vv,grains=200)

    #render.set_front([0,0,0,0.8])
    #render_triangle(*vv,fill=False)

  render.write_to_png('res/test_b_{:05d}.png'.format(i))

  i += 1


def main():

  from differentialMesh import DifferentialMesh
  from render.render import Render
  from time import time
  from modules.helpers import print_stats


  DM = DifferentialMesh(NMAX, 2*FARL, NEARL, FARL)

  DM.new_face(MID-RAD, MID,
              MID+RAD, MID,
              MID, MID+H)

  render = Render(SIZE, BACK, FRONT)
  render.set_line_width(LINEWIDTH)


  tsum = 0

  for i in xrange(10000):

    t1 = time()

    DM.optimize_position(OPT_STP)

    henum = DM.get_henum()
    rnd = 1-2*random(henum*2)
    for he in xrange(henum):

      if DM.is_surface_edge(he)>0:

        the = pi*rnd[2*he]
        rad = rnd[2*he+1]*0.5
        dx = cos(the)*rad*H
        dy = sin(the)*rad*H

        DM.new_triangle_from_surface_edge(
          he,
          H,
          dx,
          dy,
          minimum_length=H*0.8,
          maximum_length=H*2,
          merge_ragged_edge=1
        )

    DM.optimize_edges(H*2, NEARL*0.5)

    tsum += time() - t1

    if i%40==0:

      print_stats(render.num_img, tsum, DM)

      show(render, DM)

      tsum = 0


if __name__ == '__main__' :

    main()

