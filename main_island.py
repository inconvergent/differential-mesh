#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from numpy import pi
from numpy import sqrt
from numpy import zeros
from numpy import cos
from numpy import sin

MID = 0.5

NMAX = 10e7
SIZE = 5000
ONE = 1./SIZE

RAD = 3*ONE
H = sqrt(3.)*RAD
NEARL = 2*RAD
FARL = RAD*20

STP = 1./SIZE*0.5

ATTRACT_STP = STP*0.1
REJECT_STP = STP*0.1
TRIANGLE_STP = STP*0.01
ALPHA = 0
DIMINISH = 0.99

SPLIT_LIMIT = H*2
FLIP_LIMIT = NEARL*0.5

MAXIMUM_LENGTH = H*2
MINIMUM_LENGTH = H*0.8

STEPS_ITT = 100


LINEWIDTH = 1*ONE

BACK = [1,1,1,1]
FRONT = [0,0,0,0.3]

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

  render.write_to_png('res/res_isl_a_{:05d}.png'.format(i))

  i += 1


def main():

  from differentialMesh import DifferentialMesh
  from render.render import Render
  from time import time
  from modules.helpers import print_stats

  from numpy.random import randint, random

  from numpy import unique


  DM = DifferentialMesh(NMAX, 2*FARL, NEARL, FARL, PROCS)

  DM.new_faces_in_ngon(MID, MID, H, 6, 0.0)

  render = Render(SIZE, BACK, FRONT)
  render.set_line_width(LINEWIDTH)


  tsum = 0

  for i in xrange(10000):

    t1 = time()
    for _ in xrange(STEPS_ITT):

      DM.optimize_position(
        ATTRACT_STP,
        REJECT_STP,
        TRIANGLE_STP,
        ALPHA,
        DIMINISH,
        -1
      )

      henum = DM.get_henum()

      edges = unique(randint(henum,size=(henum)))
      en = len(edges)
      rnd = 1-2*random(size=en*2)
      make_island = random(size=en)>0.85

      for i,(he,isl) in enumerate(zip(edges,make_island)):

        if DM.is_surface_edge(he)>0:

          the = pi*rnd[2*i]
          rad = rnd[2*i+1]*0.5
          dx = cos(the)*rad*H
          dy = sin(the)*rad*H

          if not isl:

            DM.new_triangle_from_surface_edge(
              he,
              H,
              dx*rad*H,
              dy*rad*H,
              minimum_length=MINIMUM_LENGTH,
              maximum_length=MAXIMUM_LENGTH,
              merge_ragged_edge=1
            )

          else:

            DM.throw_seed_triangle(
              he,
              H,
              dx*rad*H,
              dy*rad*H,
              NEARL*0.5,
              the
            )

      DM.optimize_edges(
        SPLIT_LIMIT,
        FLIP_LIMIT
      )

      tsum += time() - t1

    print_stats(render.num_img, tsum, DM)
    show(render, DM)
    tsum = 0


if __name__ == '__main__' :

    main()

