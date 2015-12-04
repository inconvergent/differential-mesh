# -*- coding: utf-8 -*-


def get_surface_vertices(dm):

  res = []

  for he in xrange(dm.get_henum()):
    e = dm.is_surface_edge(he)
    if e>0:
      d = dm.get_edge_dict(he)
      res.append(d['first'])
      res.append(d['last'])

  return list(set(res))

def get_seed_selector(dm, t, sr):
  from numpy import array
  from numpy import arange
  from numpy.random import random

  if t == 'surface':

    def f(a):
      vertices = array(get_surface_vertices(dm))
      rm = (random(size=len(vertices))<sr).nonzero()[0]
      if len(rm)<1:
        return 0
      num = len(rm)
      a[:num] = vertices[rm]
      return num

  elif t == 'random':

    def f(a):
      vn = dm.get_vnum()
      vertices = arange(vn)
      rm = (random(size=vn)<sr).nonzero()[0]
      if len(rm)<1:
        return 0
      num = len(rm)
      a[:num] = vertices[rm]
      return num

  return f

def get_exporter(nmax):

  from dddUtils.ioOBJ import export_2d as export_obj
  from time import time
  from numpy import zeros

  verts = zeros((nmax, 2),'double')
  tris = zeros((nmax, 3),'int')

  t0 = time()

  def f(dm, data, itt, final=False):

    if final:
      fn = '{:s}_final.2obj'.format(data['prefix'])
    else:
      fn = '{:s}_{:010d}.2obj'.format(data['prefix'],itt)

    vnum = dm.np_get_vertices(verts)
    tnum = dm.np_get_triangles_vertices(tris)

    meta = '\n# procs {:d}\n'+\
      '# vnum {:d}\n'+\
      '# tnum {:d}\n'+\
      '# time {:f}\n'+\
      '# nearl {:f}\n'+\
      '# farl {:f}\n'+\
      '# attract stp {:f}\n'+\
      '# reject stp {:f}\n'+\
      '# triangle stp {:f}\n'+\
      '# size {:d}\n'
      
    meta = meta.format(
      data['procs'],
      vnum,
      tnum,
      time()-t0,
      data['nearl'],
      data['farl'],
      data['attract_stp'],
      data['reject_stp'],
      data['triangle_stp'],
      data['size']
   )
    export_obj(
      'mesh',
      fn, 
      verts = verts[:vnum,:], 
      faces = tris[:tnum,:], 
      meta = meta
    )

  return f

