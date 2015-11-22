#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from numpy import pi
from numpy import sqrt
from numpy import zeros
from numpy import cos
from numpy import sin
from numpy.random import random

NMAX = 10e7
SIZE = 800
ONE = 1./SIZE

RAD = 2*ONE
H = sqrt(3.)*RAD
NEARL = 2*RAD
FARL = RAD*10

OPT_STP = 1./SIZE

MID = 0.5

LINEWIDTH = 1*ONE

BACK = [1,1,1,1]
FRONT = [0,0,0,0.3]
RED = [1,0,0,0.3]
BLUE = [0,0,1,0.3]
GREEN = [0,1,0,0.3]
CYAN = [0,0.5,0.5,0.3]

STEPS_ITT = 2

NUM_SOURCES = 100

TWOPI = pi*2.

np_coord = zeros((NMAX,6), 'float')

i = 0
steps_runs = 0


def show(render, dm, sources):

  global np_coord
  global i

  render.clear_canvas()

  num = dm.np_get_triangles_coordinates(np_coord)
  render_triangle = render.triangle

  for f,vv in enumerate(np_coord[:num,:]):

    intens = dm.get_triangle_intensity(f)
    intens = intens**0.5
    rgb = [0.0]*4
    rgb[0] = intens*0.5
    rgb[1] = intens
    rgb[2] = intens
    rgb[3] = 1

    render.set_front(rgb)
    render_triangle(*vv,fill=True)

  #render.write_to_png('ani_{:05d}.png'.format(i))

  i += 1


def steps(dm):

  from time import time
  from modules.helpers import print_stats

  from numpy import unique
  from numpy.random import randint

  global steps_runs
  steps_runs += 1

  t1 = time()

  for i in xrange(STEPS_ITT):

    dm.find_nearby_sources()

    henum = dm.get_henum()

    rnd = 1-2*random(henum*2)
    for he in unique(randint(henum,size=(henum))):

      intensity = dm.get_edge_intensity(he)

      if random()<intensity and dm.is_surface_edge(he)>-1:

        the = pi*rnd[2*he]
        rad = rnd[2*he+1]*0.5
        dx = cos(the)*rad*H
        dy = sin(the)*rad*H

        dm.new_triangle_from_surface_edge(
          he,
          H,
          dx,
          dy,
          minimum_length=0,
          maximum_length=H*2,
          merge_ragged_edge=1
        )

    dm.diminish_all_vertex_intensity(0.99)

    dm.smooth_intensity(0.05)

    dm.optimize_position(OPT_STP)

    henum = dm.get_henum()

    dm.optimize_edges(2.0*H, NEARL*0.5)

    if dm.safe_vertex_positions(3*H)<0:

      print('vertices reached the boundary. stopping.')
      return False

  t2 = time()
  print_stats(steps_runs, t2-t1, dm)

  return True


def main():

  import gtk

  from differentialMesh import DifferentialMesh
  from render.render import Animate
  from modules.helpers import darts


  DM = DifferentialMesh(NMAX, 2*FARL, NEARL, FARL)

  DM.new_faces_in_ngon(MID,MID, H, 3)
  DM.set_edge_intensity(1, 1)

  sources = [(x,y) for x,y in darts(NUM_SOURCES, MID, MID, 0.40, 3*NEARL)]
  DM.initialize_sources(sources, NEARL)

  def wrap(render):

    res = steps(DM)
    show(render, DM, sources)

    return res

  render = Animate(SIZE, BACK, FRONT, wrap)
  render.set_line_width(LINEWIDTH)

  gtk.main()


if __name__ == '__main__' :

    main()

