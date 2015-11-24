#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from numpy import pi
from numpy import sqrt
from numpy import zeros

NMAX = 10e6
SIZE = 1000
ONE = 1./SIZE

RAD = 2*ONE
H = sqrt(3.)*RAD
NEARL = 2*RAD
FARL = RAD*15

OPT_STP = 1./SIZE

MID = 0.5

LINEWIDTH = ONE*1.5

BACK = [1,1,1,1]
FRONT = [0,0,0,0.3]
RED = [1,0,0,0.05]
BLUE = [0,0,1,0.3]
GREEN = [0,1,0,0.3]

STEPS_ITT = 1

TWOPI = pi*2.

np_coord = zeros((NMAX,6), 'float')
np_gen = zeros(NMAX, 'int')

i = 0
steps_runs = 0


def show(render, dm):

  global np_coord
  global np_gen
  global i

  render.clear_canvas()

  num = dm.np_get_triangles_coordinates(np_coord)
  render_random_triangle = render.random_triangle

  for f,vv in enumerate(np_coord[:num,:]):

    render.set_front(FRONT)
    render_random_triangle(*vv,grains=60)

  #render.write_to_png('ani_{:05d}.png'.format(i))

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

    dm.optimize_position(OPT_STP)

    henum = dm.get_henum()

    edges = unique(randint(henum,size=(henum)))
    en = len(edges)
    rnd = 1-2*random(en*2)
    make_island = random(size=en)>0.95

    for i,(he,isl) in enumerate(zip(edges,make_island)):

      if dm.is_surface_edge(he)>0:

        the = pi*rnd[2*i]
        rad = rnd[2*i+1]*0.5

        if not isl:

          dx = cos(the)*rad*H
          dy = sin(the)*rad*H
          dm.new_triangle_from_surface_edge(
            he,
            H,
            dx,
            dy,
            minimum_length=H*0.8,
            maximum_length=H*2,
            merge_ragged_edge=1
          )

        else:

          dx = cos(the)*rad*H
          dy = sin(the)*rad*H
          dm.throw_seed_triangle(
            he,
            H,
            dx,
            dy,
            NEARL*0.5
          )

    dm.optimize_edges(2.0*H, NEARL*0.5)

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

  DM = DifferentialMesh(NMAX, 2*FARL, NEARL, FARL)

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

