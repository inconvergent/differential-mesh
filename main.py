#!/usr/bin/python3
# -*- coding: utf-8 -*-



from numpy import pi
from numpy import sqrt
from numpy import zeros
from numpy import cos
from numpy import sin
from numpy.random import random

from fn.fn import Fn

from modules.timers import named_sub_timers

NMAX = 10e7
SIZE = 2000
ONE = 1./SIZE

RAD = 3*ONE
H = sqrt(3.)*RAD
NEARL = 2*RAD
FARL = RAD*20

OPT_STP = 1./SIZE*0.5

MID = 0.5

STEPS_ITT = 10

LINEWIDTH = 1*ONE

STP = 1./SIZE*0.5

ATTRACT_STP = STP*0.1
REJECT_STP = STP
TRIANGLE_STP = STP*0.01
ALPHA = 0
DIMINISH = 0.99


MINIMUM_LENGTH = H*0.8
MAXIMUM_LENGTH = H*2

SPLIT_LIMIT = H*2
FLIP_LIMIT = NEARL*0.5

BACK = [1,1,1,1]
FRONT = [0,0,0,0.3]

PROCS = 6


TWOPI = pi*2.

np_coord = zeros((NMAX,6), 'float')


def show(render, dm, fn):

  global np_coord

  render.clear_canvas()

  num = dm.np_get_triangles_coordinates(np_coord)
  render_random_triangle = render.random_triangle
  render_triangle = render.triangle
  rgba = FRONT
  render.set_front(rgba)

  for e,vv in enumerate(np_coord[:num,:]):

    # render_random_triangle(*vv,grains=80)
    render_triangle(*vv)

  render.write_to_png(fn)


def main():

  from differentialMesh import DifferentialMesh
  from iutils.render import Render
  from time import time
  from modules.helpers import print_stats
  from numpy import array

  from modules.utils import get_exporter
  exporter = get_exporter(NMAX)

  DM = DifferentialMesh(NMAX, 2*FARL, NEARL, FARL, PROCS)

  DM.new_faces_in_ngon(MID, MID, H, 6, 0.0)

  render = Render(SIZE, BACK, FRONT)
  render.set_line_width(LINEWIDTH)

  fn = Fn(prefix='./res/')


  # st = named_sub_timers()

  tsum = 0


  for i in range(10000000):

    t1 = time()
    for _ in range(STEPS_ITT):

      # st.start()
      DM.optimize_position(
        ATTRACT_STP,
        REJECT_STP,
        TRIANGLE_STP,
        ALPHA,
        DIMINISH,
        -1
      )
      # st.t('opt')

      henum = DM.get_henum()
      # st.t('rnd')

      surface_edges = array(
        [DM.is_surface_edge(e)>0
        for e in range(henum)],
        'bool').nonzero()[0]

      # st.t('surf')

      rnd = random(size=len(surface_edges)*2)
      the = (1.-2*rnd[::2])*pi
      rad = rnd[1::2]*0.5*H
      dx = cos(the)*rad
      dy = sin(the)*rad
      # st.t('rnd2')

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
      # st.t('tri')

      # st.start()
      DM.optimize_edges(
        SPLIT_LIMIT,
        FLIP_LIMIT
      )
      # st.t('opte')

      tsum += time() - t1

    print_stats(i*STEPS_ITT, tsum, DM)
    name = fn.name()

    ## export png
    show(render, DM, name+'.png')

    ## export obj
    exporter(
      DM,
      name + '.2obj',
      {
        'procs': PROCS,
        'nearl': NEARL,
        'farl': FARL,
        'reject_stp': 0,
        'attract_stp': 0,
        'triangle_stp': 0,
        'size': SIZE
      },
      i*STEPS_ITT
    )
    tsum = 0
    # st.p()



if __name__ == '__main__' :

    main()

