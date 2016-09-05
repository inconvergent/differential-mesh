#!/usr/bin/python3
# -*- coding: utf-8 -*-



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

STEPS_ITT = 100

STP = 1./SIZE

ATTRACT_STP = STP*0.1
REJECT_STP = STP
TRIANGLE_STP = STP*0.01
ALPHA = 0
DIMINISH = 0.99


MINIMUM_LENGTH = H*0.8
MAXIMUM_LENGTH = H*2

SPLIT_LIMIT = H*2
FLIP_LIMIT = NEARL*0.5


BACK = [1, 1.0, 1.0, 1]
FRONT = [0,0,0,0.3]

RED = [1,0,0,0.3]
BLUE = [0,0,1,0.3]
GREEN = [0,1,0,0.3]
CYAN = [0,0.5,0.5,0.5]

NUM_SOURCES = 2500*4

PROCS = 6


GRAINS = 80

np_coord = zeros((NMAX,6), 'float')
np_edges_coord = zeros((NMAX,4), 'float')
np_gen = zeros(NMAX, 'int')


def show_circles(render, dm, sources):

  global np_coord
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

  render.write_to_png('res/ello_circ_i_{:05d}.png'.format(render.num_img))


def show_triangles(render, dm, sources):

  global np_coord

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


def main():

  from time import time
  from iutils.render import Render
  from differentialMesh import DifferentialMesh
  from modules.helpers import darts
  from modules.helpers import print_stats
  from numpy import array


  DM = DifferentialMesh(NMAX, 2*FARL, NEARL, FARL, PROCS)

  DM.new_faces_in_ngon(MID,MID, H, 7, 0)
  DM.set_edge_intensity(1, 1)

  sources = [(x,y) for x,y in darts(NUM_SOURCES, MID, MID, 0.43, 0.002)]
  DM.initialize_sources(sources, NEARL)

  render = Render(SIZE, BACK, FRONT)


  for i in range(1000000):

    t1 = time()
    for _ in range(STEPS_ITT):

      DM.find_nearby_sources()

      henum = DM.get_henum()

      surface_edges = array(
        [DM.is_surface_edge(e)>0 and r<DM.get_edge_intensity(e)
        for e,r in enumerate(random(size=henum))],
        'bool').nonzero()[0]

      rnd = random(size=len(surface_edges)*2)
      the = (1.-2*rnd[::2])*pi
      rad = rnd[1::2]*0.5*H
      dx = cos(the)*rad
      dy = sin(the)*rad

      DM.new_triangles_from_surface_edges(
        surface_edges,
        len(surface_edges),
        H,
        dx,
        dy,
        MINIMUM_LENGTH,
        MAXIMUM_LENGTH,
        1
      )

      DM.optimize_position(
        ATTRACT_STP,
        REJECT_STP,
        TRIANGLE_STP,
        ALPHA,
        DIMINISH,
        -1
      )

      henum = DM.get_henum()

      DM.optimize_edges(
        SPLIT_LIMIT,
        FLIP_LIMIT
      )

      if DM.safe_vertex_positions(3*H)<0:

        show_circles(render, DM, sources)
        print('vertices reached the boundary. stopping.')
        return

    show_circles(render, DM, sources)

    t2 = time()
    print_stats(i*STEPS_ITT, t2-t1, DM)


if __name__ == '__main__' :

    main()

