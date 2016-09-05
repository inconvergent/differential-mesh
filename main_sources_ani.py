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
SIZE = 1000
ONE = 1./SIZE

RAD = 2*ONE
H = sqrt(3.)*RAD
NEARL = 2*RAD
FARL = RAD*10

MID = 0.5

LINEWIDTH = 1*ONE

STP = 1./SIZE
ATTRACT_STP = STP*0.1
REJECT_STP = STP*0.1
TRIANGLE_STP = STP*0.01
ALPHA = 0.05
DIMINISH = 0.99

MINIMUM_LENGTH = H*0.8
MAXIMUM_LENGTH = H*2

SPLIT_LIMIT = H*2
FLIP_LIMIT = NEARL*0.5

BACK = [1,1,1,1]
FRONT = [0,0,0,0.3]

NUM_SOURCES = 100

TWOPI = pi*2.

PROCS = 2

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

  from numpy import array


  global steps_runs
  steps_runs += 1

  t1 = time()

  dm.find_nearby_sources()

  henum = dm.get_henum()

  surface_edges = array(
    [dm.is_surface_edge(i)>0 and r<dm.get_edge_intensity(i)
    for i,r in enumerate(random(size=henum))],
    'bool').nonzero()[0]

  rnd = random(len(surface_edges)*2)
  the = (1.-2*rnd[::2])*pi
  rad = rnd[1::2]*0.5*H
  dx = cos(the)*rad
  dy = sin(the)*rad

  dm.new_triangles_from_surface_edges(
    surface_edges,
    len(surface_edges),
    H,
    dx,
    dy,
    MINIMUM_LENGTH,
    MAXIMUM_LENGTH,
    1
  )

  dm.optimize_position(
    ATTRACT_STP,
    REJECT_STP,
    TRIANGLE_STP,
    ALPHA,
    DIMINISH,
    1
  )

  henum = dm.get_henum()

  # dm.optimize_edges(2.0*H, NEARL*0.5)
  dm.optimize_edges(SPLIT_LIMIT, FLIP_LIMIT)

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


  DM = DifferentialMesh(NMAX, 2*FARL, NEARL, FARL, PROCS)

  DM.new_faces_in_ngon(MID,MID, H, 3, 0)
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

