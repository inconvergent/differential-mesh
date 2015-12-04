#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from numpy import pi
from numpy import sqrt
from numpy import zeros

NMAX = 10e6
SIZE = 520
ONE = 1./SIZE

RAD = 1.5*ONE
H = sqrt(3.)*RAD
NEARL = 2*RAD
FARL = RAD*15

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

MID = 0.5

LINEWIDTH = ONE*1.5

BACK = [1,1,1,1]
FRONT = [0,0,0,0.3]

STEPS_ITT = 1

TWOPI = pi*2.

np_coord = zeros((NMAX,6), 'float')
np_gen = zeros(NMAX, 'int')

i = 0
steps_runs = 0

PROCS = 2


def show(render, dm):

  global np_coord
  global np_gen
  global i

  render.clear_canvas()

  num = dm.np_get_triangles_coordinates(np_coord)
  render_random_triangle = render.random_triangle

  for f,vv in enumerate(np_coord[:num,:]):

    render.set_front(FRONT)
    render_random_triangle(*vv,grains=45)

  # render.write_to_png('ani_{:05d}.png'.format(i))

  i += 1


def steps(dm):

  from numpy import cos
  from numpy import sin
  from numpy import unique
  from numpy.random import randint, random
  from time import time
  from modules.helpers import print_stats

  global steps_runs
  steps_runs += 1

  t1 = time()
  for i in xrange(STEPS_ITT):

    dm.optimize_position(
      ATTRACT_STP,
      REJECT_STP,
      TRIANGLE_STP,
      ALPHA,
      DIMINISH,
      -1
    )

    henum = dm.get_henum()

    edges = unique(randint(henum,size=(henum)))
    en = len(edges)
    rnd = 1-2*random(en*2)
    make_island = random(size=en)>0.85

    for i,(he,isl) in enumerate(zip(edges,make_island)):

      if dm.is_surface_edge(he)>0:

        the = pi*rnd[2*i]
        rad = rnd[2*i+1]*0.5
        dx = cos(the)
        dy = sin(the)

        if not isl:

          dm.new_triangle_from_surface_edge(
            he,
            H,
            dx*rad*H,
            dy*rad*H,
            minimum_length=MINIMUM_LENGTH,
            maximum_length=MAXIMUM_LENGTH,
            merge_ragged_edge=1
          )

        else:

          dm.throw_seed_triangle(
            he,
            H,
            dx*rad*H,
            dy*rad*H,
            NEARL*0.5,
            the
          )

    dm.optimize_edges(
      SPLIT_LIMIT,
      FLIP_LIMIT
    )

    if dm.safe_vertex_positions(3*H)<0:

      print('vertices reached the boundary. stopping.')
      return False

  t2 = time()
  print_stats(steps_runs, t2-t1, dm)

  return True

i = 0


def main():

  import gtk

  from differentialMesh import DifferentialMesh
  from render.render import Animate

  DM = DifferentialMesh(NMAX, 2*FARL, NEARL, FARL, PROCS)

  DM.new_faces_in_ngon(MID,MID, H, 6, 0.0)

  def wrap(render):

    res = steps(DM)
    show(render, DM)

    return res

  render = Animate(SIZE, BACK, FRONT, wrap)
  render.set_line_width(LINEWIDTH)

  gtk.main()


if __name__ == '__main__' :

    main()

