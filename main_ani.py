#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from numpy import pi
from numpy import sqrt
from numpy import zeros
from numpy import cos
from numpy import sin
from numpy.random import random

NMAX = 10e6
SIZE = 800
ONE = 1./SIZE

RAD = 6*ONE
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

PROCS = 2

np_coord = zeros((NMAX,6), 'float')
np_gen = zeros(NMAX, 'int')

i = 0
steps_runs = 0


def show(render, dm):

  from numpy import floor

  global np_coord
  global np_gen
  global i

  colors = [
    #[0,0,0,0.8],
    #[0.4,0.4,0.4,0.1],
    #[0.5,0.5,0.5,0.1],
    [0.6,0.6,0.6,0.1]
  ]

  render.clear_canvas()

  num = dm.np_get_triangles_coordinates(np_coord)
  #dm.np_get_triangles_gen(np_gen)
  render_random_triangle = render.random_triangle
  render_triangle = render.triangle
  render_circle = render.circle

  #np_gen[:num] = floor(np_gen[:num]/4.)%len(colors)

  for f,vv in enumerate(np_coord[:num,:]):

    #render.set_front(colors[np_gen[f]])
    render.set_front(FRONT)
    render_triangle(*vv,fill=False)
    #render_random_triangle(*vv,grains=60)

    #render.set_front([1,1,1,0.05])
    #render.set_front(FRONT)
    #render_triangle(*vv,fill=False)

    #render_random_triangle(*vv,grains=80)

    rad = ONE*3
    render.set_front([0,0.5,0.5,0.3])
    render_circle(vv[0], vv[1], rad, fill=True)
    render_circle(vv[2], vv[3], rad, fill=True)
    render_circle(vv[4], vv[5], rad, fill=True)

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

    dm.optimize_position(OPT_STP)

    #various tests. some of these will tend to fail. especially when
    #you use optimize_position

    #for f in xrange(dm.get_fnum()):

      #if dm.triangle_integrity(f)<0:
        #raise ValueError('integrity 1')

      ## if dm.triangle_rotation(f)<0:
        ## raise ValueError('rotation 1')

    #for e in xrange(dm.get_henum()):

      #if dm.edge_integrity(e)<0:
        #raise ValueError('edge integrity 1')

    henum = dm.get_henum()
    rnd = 1-2*random(henum*2)

    edges = unique(randint(henum,size=(henum)))

    for he in edges:

      if dm.is_surface_edge(he)>0:

        the = pi*rnd[2*he]
        rad = rnd[2*he+1]*0.5
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

  DM = DifferentialMesh(NMAX, 2*FARL, NEARL, FARL, PROCS)

  DM.new_faces_in_ngon(MID,MID, H, 6, 0.0)

  def wrap(render):

    res = steps(DM)
    show(render, DM)

    return res

  render = Animate(SIZE, BACK, FRONT, wrap)
  # render.get_colors_from_file('../colors/red_earth.gif')
  render.set_line_width(LINEWIDTH)

  gtk.main()


if __name__ == '__main__' :

    main()

