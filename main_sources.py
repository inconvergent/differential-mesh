#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from numpy import pi
from numpy import sqrt
from numpy import zeros
from numpy import cos
from numpy import sin
from numpy import power
from numpy import floor
from numpy.random import random

TWOPI = pi*2.
MID = 0.5


NMAX = 10e7
SIZE = 10000
ONE = 1./SIZE*0.5

RAD = 5*ONE
H = sqrt(3.)*RAD
NEARL = 2*RAD
FARL = RAD*13

OPT_STP = 1./SIZE


BACK = [1, 1.0, 1.0, 1]
FRONT = [0,0,0,0.3]

RED = [1,0,0,0.3]
BLUE = [0,0,1,0.3]
GREEN = [0,1,0,0.3]
CYAN = [0,0.5,0.5,0.5]

STEPS_ITT = 200

NUM_SOURCES = 2500*4

PROCS = 6


GRAINS = 80

np_coord = zeros((NMAX,6), 'float')
np_edges_coord = zeros((NMAX,4), 'float')
np_gen = zeros(NMAX, 'int')

i = 0

def show_circles(render, dm, sources):

  global np_coord
  global i
  global np_gen

  render.clear_canvas()

  #num = dm.np_get_edges_coordinates(np_edges_coord)
  #render_random_circle = render.random_circle
  render_random_triangle = render.random_triangle
  #render_random_uniform_circle = render.random_uniform_circle
  #render_line = render.line

  render.set_front(FRONT)
  render.set_line_width(render.pix*2)

  #rotation = 10

  num = dm.np_get_triangles_coordinates(np_coord)
  render_triangle = render.triangle

  for f,vv in enumerate(np_coord[:num,:]):

    render.set_front([0.,0.,0.,0.1])
    render_random_triangle(*vv,grains=200)

  #for e,vv in enumerate(np_edges_coord[:num,:]):

    ##g = (np_gen[e]%rotation)/float(rotation)
    ##rgba = [g]*4
    ##rgba[3] = 0.4

    ##if g<=0.:
      ##rgba = RED
      ##render.set_front(rgba)
      ##render_line(*vv)
    ##else:
      ##rgba = FRONT
      ##render.set_front(rgba)
    #render_random_uniform_circle(vv[0], vv[1], dd[e], grains=30, dst=0)


  render.write_to_png('res/ello_circ_i_{:05d}.png'.format(i))

  i += 1


def show_triangles(render, dm, sources):

  global np_coord
  global i

  render.clear_canvas()

  #render_circle = render.circle

  #source_rad = ONE*10
  #for x,y in sources:

    #render.set_front(CYAN)
    #render_circle(x, y, source_rad, fill=True)

  num = dm.np_get_triangles_coordinates(np_coord)
  render_random_triangle = render.random_triangle
  #render_triangle = render.triangle

  for f,vv in enumerate(np_coord[:num,:]):

    intens = dm.get_triangle_intensity(f)
    rgb = [0.05]*4
    rgb[1] += intens
    rgb[2] += intens
    rgb[3] = 0.4

    render.set_front(rgb)
    render_random_triangle(*vv,grains=GRAINS)

  render.write_to_png('res/expand_c_{:05d}.png'.format(i))

  i += 1


def main():

  from render.render import Render
  from differentialMesh import DifferentialMesh
  from modules.helpers import darts
  from modules.helpers import print_stats

  from time import time

  from numpy import unique
  from numpy.random import randint


  DM = DifferentialMesh(NMAX, 2*FARL, NEARL, FARL, PROCS)

  DM.new_faces_in_ngon(MID,MID, H, 7, 0)
  DM.set_edge_intensity(1, 1)

  sources = [(x,y) for x,y in darts(NUM_SOURCES, MID, MID, 0.43, 0.002)]
  DM.initialize_sources(sources, NEARL)

  render = Render(SIZE, BACK, FRONT)


  for i in xrange(1000000):

    t1 = time()
    for i in xrange(STEPS_ITT):

      DM.find_nearby_sources()

      henum = DM.get_henum()

      rnd = 1-2*random(henum*2)
      for he in unique(randint(henum,size=(henum))):

        intensity = DM.get_edge_intensity(he)

        if random()<intensity and DM.is_surface_edge(he)>-1:

          the = pi*rnd[2*he]
          rad = rnd[2*he+1]*0.5
          dx = cos(the)*rad*H
          dy = sin(the)*rad*H

          DM.new_triangle_from_surface_edge(
            he,
            H,
            dx,
            dy,
            minimum_length=0,
            maximum_length=H*2,
            merge_ragged_edge=1
          )

      DM.diminish_all_vertex_intensity(0.99)

      DM.smooth_intensity(0.05)

      DM.optimize_position(OPT_STP)

      henum = DM.get_henum()

      DM.optimize_edges(2.0*H, NEARL*0.5)

      if DM.safe_vertex_positions(3*H)<0:

        show_circles(render, DM, sources)
        print('vertices reached the boundary. stopping.')
        return

    show_circles(render, DM, sources)

    t2 = time()
    print_stats(render.num_img*STEPS_ITT, t2-t1, DM)


if __name__ == '__main__' :

    main()

