# -*- coding: utf-8 -*-
# cython: profile=True

from __future__ import division

cimport cython
cimport mesh

from zonemap cimport Zonemap

from cython.parallel import parallel, prange

from numpy import pi

from libc.math cimport sqrt
from libc.math cimport pow
from libc.math cimport cos
from libc.math cimport sin

from helpers cimport double_array_init
from helpers cimport long_array_init
from helpers cimport vcross

import numpy as np
cimport numpy as np



cdef double TWOPI = 2.*pi


cdef class DifferentialMesh(mesh.Mesh):

  def __init__(self, long nmax, double zonewidth, double nearl, double farl, long procs):

    mesh.Mesh.__init__(self, nmax, zonewidth, procs)

    """
    - nearl is the closest comfortable distance between two vertices.

    - farl is the distance beyond which disconnected vertices will ignore
    each other
    """

    self.nearl = nearl

    self.farl = farl

    self.num_sources = 0

    self.source_zonemap = Zonemap(self.nz)
    self.source_zonemap.__assign_xy_arrays(self.SX, self.SY)

    print('nearl: {:f}'.format(nearl))
    print('farl: {:f}'.format(farl))

    return

  def __cinit__(self, long nmax, *arg, **args):

    self.DX = <double *>malloc(nmax*sizeof(double))

    self.DY = <double *>malloc(nmax*sizeof(double))

    self.SX = <double *>malloc(nmax*sizeof(double))

    self.SY = <double *>malloc(nmax*sizeof(double))

    return

  def __dealloc__(self):

    free(self.DX)

    free(self.DY)

    free(self.SX)

    free(self.SY)

    return

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  @cython.cdivision(True)
  cdef long __reject(
    self,
    long v1,
    long *vertices,
    long num,
    double scale,
    double *diffx,
    double *diffy
  ) nogil:
    """
    all vertices will move away from all neighboring (closer than farl)
    vertices
    """

    cdef long k
    cdef long neigh

    cdef double dx
    cdef double dy
    cdef double nrm
    cdef double s

    cdef double resx = 0.
    cdef double resy = 0.

    for k in range(num):

      neigh = vertices[k]

      if neigh == v1:
        continue

      dx = self.X[v1]-self.X[neigh]
      dy = self.Y[v1]-self.Y[neigh]
      nrm = sqrt(dx*dx+dy*dy)

      if nrm>self.farl or nrm<=1e-9:
        continue

      s = self.farl/nrm-1.0

      # if nrm<self.nearl:
        # s = s*2.

      resx += dx*s
      resy += dy*s

    diffx[v1] += resx*scale
    diffy[v1] += resy*scale

    return 1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  @cython.cdivision(True)
  cdef long __triangle(
    self,
    long v1,
    double *diffx,
    double *diffy,
    double scale,
    long *vertices,
    long num
  ) nogil:

    cdef long k

    for k in xrange(num):

      self.__edge_vertex_force(v1, diffx, diffy, vertices[k], scale)

    return num

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  @cython.cdivision(True)
  cdef long __edge_vertex_force(
    self,
    long v1,
    double *diffx,
    double *diffy,
    long he1,
    double scale
  ) nogil:

    cdef long henum = self.henum

    cdef long a = self.HE[he1].first
    cdef long b = self.HE[he1].last

    cdef double x = (self.X[b]+self.X[a])*0.5
    cdef double y = (self.Y[b]+self.Y[a])*0.5

    cdef double dx = self.X[v1]-x
    cdef double dy = self.Y[v1]-y

    cdef double nrm = sqrt(dx*dx+dy*dy)

    if nrm<=0:

      return -1

    if nrm>0.5*sqrt(3.0)*self.nearl:

      #pass
      diffx[v1] += -dx/nrm*scale
      diffy[v1] += -dy/nrm*scale

    else:

      diffx[v1] += dx/nrm*scale
      diffy[v1] += dy/nrm*scale

    return 1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  @cython.cdivision(True)
  cdef long __attract(
    self,
    long v1,
    long *connected,
    long num,
    double scale,
    double *diffx,
    double *diffy
  ) nogil:
    """
    vertices will move towards all connected vertices further away than
    nearl
    """

    cdef long v2
    cdef long k

    cdef double dx
    cdef double dy
    cdef double nrm

    cdef double s

    for k in xrange(num):

      v2 = connected[k]

      dx = self.X[v2]-self.X[v1]
      dy = self.Y[v2]-self.Y[v1]
      nrm = sqrt(dx*dx+dy*dy)

      if nrm>self.nearl:

        ## attract
        s = scale/nrm
        diffx[v1] += dx*s
        diffy[v1] += dy*s

    return 1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cdef long __optimize_position(
    self,
    double attract_scale,
    double reject_scale,
    double triangle_scale,
    double alpha,
    double diminish,
    long scale_intensity
  ) nogil:

    cdef long asize = self.zonemap.__get_max_sphere_count()
    cdef long *vertices
    cdef long *connected
    cdef long num
    cdef long cnum
    cdef long v
    cdef long i

    cdef double x
    cdef double y
    cdef double dx
    cdef double dy
    cdef double nrm
    cdef double stp_limit = self.nearl*0.3

    with nogil, parallel(num_threads=self.procs):

      vertices = <long *>malloc(asize*sizeof(long))
      connected = <long *>malloc(asize*sizeof(long))

      for v in prange(self.vnum, schedule='guided'):

        self.DX[v] = 0.0
        self.DY[v] = 0.0

        cnum = self.__get_connected_vertices(v, connected)
        num = self.zonemap.__sphere_vertices(
          self.X[v],
          self.Y[v],
          self.farl,
          vertices
        )
        self.__reject(
          v,
          vertices,
          num,
          reject_scale,
          self.DX,
          self.DY
        )

        self.__attract(
          v,
          connected,
          cnum,
          attract_scale,
          self.DX,
          self.DY
        )
        self.__smooth_intensity(
          v,
          alpha,
          self.I,
          self.DI,
          connected,
          cnum
        )

        cnum = self.__get_opposite_edges(v, connected)
        self.__triangle(
          v,
          self.DX,
          self.DY,
          triangle_scale,
          connected,
          cnum
        )

      free(vertices)
      free(connected)

      for v in prange(self.vnum, schedule='static'):

        dx = self.DX[v]
        dy = self.DY[v]

        nrm = sqrt(dx*dx+dy*dy)

        if nrm>stp_limit:
          dx = dx / nrm * stp_limit
          dy = dy / nrm * stp_limit

        if scale_intensity>0:
          x = self.X[v] + self.I[v]*dx
          y = self.Y[v] + self.I[v]*dy
        else:
          x = self.X[v] + dx
          y = self.Y[v] + dy

        self.X[v] = x
        self.Y[v] = y
        self.I[v] = self.DI[v]*diminish

    with nogil:
      for v in xrange(self.vnum):
        self.zonemap.__update_v(v)

    return 1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cpdef long initialize_sources(self, list sources, double source_rad):

    cdef long i
    cdef long num_sources
    cdef double x
    cdef double y

    num_sources = len(sources)
    self.num_sources = num_sources
    self.source_rad = source_rad

    for i in xrange(num_sources):

      x,y = sources[i]
      self.SX[i] = x
      self.SY[i] = y

      self.source_zonemap.__add_vertex(i)

    print('initialized sources: {:d}'.format(num_sources))

    return 1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cpdef long find_nearby_sources(self, long hit_limit=10):

    return self.__find_nearby_sources(hit_limit)


  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  @cython.cdivision(True)
  cdef long __merge_ragged_edge(self, long he1, long he3, long he4) nogil:

    cdef double *normal1 = [-1,-1]
    cdef double *normal2 = [-1,-1]

    cdef double dot
    cdef long s3
    cdef long s4
    cdef long ss
    cdef long f
    cdef long ne
    cdef long normal_test1
    cdef long normal_test2

    cdef long n = 0

    cdef double vc = -1

    ## clockwise / forward surface
    cdef long s = self.__next_surface(he1, -1)

    if s>1:

      ## TODO: there is some duplication of the rotational checks here.

      normal_test1 = self.__get_surface_edge_outward_normal(he4, normal1)
      normal_test2 = self.__get_surface_edge_outward_normal(s, normal2)
      dot = normal1[0]*normal2[0] + normal1[1]*normal2[1]

      vc = vcross(
        self.X[self.HE[he4].first], self.Y[self.HE[he4].first],
        self.X[self.HE[he4].last], self.Y[self.HE[he4].last],
        self.X[self.HE[s].last], self.Y[self.HE[s].last]
      )

      if dot<=0 and vc<0 and normal_test1>0 and normal_test2>0:

        ss = self.__new_edge(self.HE[s].last, self.HE[s].first)
        self.__set_mutual_twins(ss,s)

        s4 = self.__new_edge(self.HE[he4].last, self.HE[he4].first)
        self.__set_mutual_twins(he4,s4)

        ne = self.__new_edge(self.HE[he4].first, self.HE[s].last)

        self.__set_next_of_triangle(ss,s4,ne)

        f = self.__new_face(ne)
        self.__set_face_of_three_edges(f, ss, s4, ne)
        self.__set_gen_of_three_edges(self.HE[he4].gen, ss, s4, ne)

        n += 1

    ## clockwise / forward surface
    s = self.__next_surface(he1, 1)

    if s>1:

      ## TODO: there is some duplication of the rotational checks here.

      normal_test1 = self.__get_surface_edge_outward_normal(he3, normal1)
      normal_test2 = self.__get_surface_edge_outward_normal(s, normal2)
      dot = normal1[0]*normal2[0] + normal1[1]*normal2[1]

      vc = vcross(
        self.X[self.HE[he3].first], self.Y[self.HE[he3].first],
        self.X[self.HE[he3].last], self.Y[self.HE[he3].last],
        self.X[self.HE[s].first], self.Y[self.HE[s].first]
      )

      if dot<=0 and vc<0 and normal_test1>0 and normal_test2>0:

        ss = self.__new_edge(self.HE[s].last, self.HE[s].first)
        self.__set_mutual_twins(ss,s)

        s3 = self.__new_edge(self.HE[he3].last, self.HE[he3].first)
        self.__set_mutual_twins(s3,he3)

        ne = self.__new_edge(self.HE[s].first, self.HE[he3].last)

        self.__set_next_of_triangle(ss,ne,s3)

        f = self.__new_face(ne)
        self.__set_face_of_three_edges(f, ss, s3, ne)
        self.__set_gen_of_three_edges(self.HE[he3].gen, ss, s3, ne)

        n += 1

    return n

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cdef long __find_nearby_sources(self, long hit_limit):

    cdef long v
    cdef long n
    cdef long d
    cdef long *sources

    cdef long num

    to_delete = <long *>malloc(hit_limit*self.vnum*sizeof(long))
    long_array_init(to_delete, hit_limit*self.vnum, -1)
    counts = <long *>malloc(self.vnum*sizeof(long))

    cdef long hits = 0

    with nogil, parallel(num_threads=self.procs):

      sources = <long *>malloc(hit_limit*sizeof(long))

      for v in prange(self.vnum, schedule='guided'):

        num = self.source_zonemap.__sphere_vertices(
          self.X[v],
          self.Y[v],
          self.source_rad,
          sources
        )

        counts[v] = num

        if num>0:
          # print(num)
          self.__set_vertex_intensity(v, 1.0)
          for n in xrange(num):
            to_delete[v*hit_limit+n] = sources[n]
          hits += num

      # print(hits)

      free(sources)

    cdef set deleted = set()

    for v in xrange(self.vnum):
      for n in xrange(counts[v]):
        d = to_delete[v*hit_limit+n]

        if not d in deleted:
          self.source_zonemap.__del_vertex(d)
          deleted.add(d)

    # print(deleted)

    free(to_delete)
    free(counts)

    return hits

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  @cython.cdivision(True)
  cdef long __smooth_intensity(
    self,
    long v1,
    double alpha,
    double *old,
    double *new,
    long *vertices,
    long num
  ) nogil:

    cdef double intens = old[v1]

    for k in xrange(num):

      intens = intens + alpha*old[vertices[k]]
      #new[v1] += ((b-a)*alpha + a*(1.0-alpha))/(1.0-alpha)

    new[v1] = intens/(1.0+<double>(num*alpha))

    return 1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  @cython.cdivision(True)
  cdef long __throw_seed_triangle(
    self,
    long he1,
    double h,
    double dx,
    double dy,
    double rad,
    double rot
  ) nogil:

    if self.__is_surface_edge(he1)<0:

      return -1

    if self.__triangle_rotation(self.HE[he1].face)<0:

      return -1

    cdef double *normal = [-1,-1]

    if self.__get_surface_edge_outward_normal(he1, normal)<0:

      return -1

    cdef long a = self.HE[he1].first
    cdef long b = self.HE[he1].last

    cdef double x1 = (self.X[a]+self.X[b])*0.5 + normal[0]*h + dx
    cdef double y1 = (self.Y[a]+self.Y[b])*0.5 + normal[1]*h + dy

    if self.zonemap.__sphere_is_free(x1, y1, h)<0:

      return -1

    # return self.__new_faces_in_ngon(x1,y1,rad,3,rot)
    return self.__new_face_in_triangle(x1,y1,rad,rot)

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  @cython.cdivision(True)
  cdef long __new_triangle_from_surface_edge(
    self,
    long he1,
    double h,
    double dx,
    double dy,
    double minimum_length,
    double maximum_length,
    long merge_ragged_edge
  ) nogil:

    """
    creates a new triangle from edge he1 by adding one vertex and two edges.

    -- he1 is the face from which to attempt to grow a new triangle
    -- h is the height of the new triangle
    -- dx is noise in the x direction added to the position of the new vertex
    -- dy is noise in the y direction added to the position of the new vertex
    -- if merge_ragged_edge>0 we will attempt to connect the new vertex to the
         structure by adding a one or two new edges (triangles).
    """

    if self.__is_surface_edge(he1)<0:

      return -1

    if self.__triangle_rotation(self.HE[he1].face)<0:

      return -1

    cdef double *normal = [-1,-1]

    if self.__get_surface_edge_outward_normal(he1, normal)<0:

      return -1

    cdef double length = self.__get_edge_length(he1)

    if length<minimum_length or length>maximum_length:

      return -1

    cdef long first = self.HE[he1].first
    cdef long last = self.HE[he1].last

    cdef double x1 = (self.X[first] + self.X[last])*0.5 + normal[0]*h + dx
    cdef double y1 = (self.Y[first] + self.Y[last])*0.5 + normal[1]*h + dy

    cdef double vc = vcross(
      self.X[self.HE[he1].first], self.Y[self.HE[he1].first],
      self.X[self.HE[he1].last], self.Y[self.HE[he1].last],
      x1, y1
    )

    if vc>0:

      return 1

    if self.zonemap.__sphere_is_free(x1, y1, h)<0:

      return -1

    cdef long v1 = self.__new_vertex(x1, y1)

    self.__set_vertex_intensity(v1, (self.I[self.HE[he1].first] +
                                     self.I[self.HE[he1].last])*0.5)

    cdef long he2 = self.__new_edge(last, first)
    self.__set_mutual_twins(he2, he1)
    cdef long he3 = self.__new_edge_from_edge(he2, v1)
    cdef long he4 = self.__new_edge_from_edge(he3, last)
    self.HE[he4].next = he2

    self.__set_edge_of_vertex(v1,he4)

    cdef long f = self.__new_face(he2)

    self.__set_face_of_three_edges(f, he2, he3, he4)
    self.__set_gen_of_three_edges(self.HE[he1].gen+1, he2, he3, he4)

    if merge_ragged_edge>0:

      self.__merge_ragged_edge(he1, he3, he4)

    return 1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  @cython.cdivision(True)
  cdef long __new_triangles_from_surface_edges(
    self,
    long[:] edges,
    long num,
    double h,
    double[:] dx,
    double[:] dy,
    double minimum_length,
    double maximum_length,
    long merge_ragged_edge
  ):

    """
    attempts to create new triangles from edges by adding one vertex and two edges.

    -- edges is the faces from which to attempt to grow a new triangle
    -- h is the height of the new triangle
    -- dx is noise in the x direction added to the position of the new vertex
    -- dy is noise in the y direction added to the position of the new vertex
    -- if merge_ragged_edge>0 we will attempt to connect the new vertex to the
         structure by adding a one or two new edges (triangles).
    """

    cdef double *ok = <double*>malloc(num*sizeof(double))
    cdef double *normal
    cdef double *xys = <double*>malloc(2*num*sizeof(double))
    cdef long he1
    cdef long first
    cdef long last
    cdef long i

    cdef double length
    cdef double x1
    cdef double y1

    with nogil, parallel(num_threads=self.procs):
    # if True:

      normal = <double*>malloc(2*sizeof(double))
      for i in prange(num, schedule='guided'):

        he1 = edges[i]

        if self.__is_surface_edge(he1)<0:
          ok[i] = -1
          # # print('surf')
          continue

        if self.__triangle_rotation(self.HE[he1].face)<0:
          ok[i] = -1
          # print('rot')
          continue

        if self.__get_surface_edge_outward_normal(he1, normal)<0:
          ok[i] = -1
          # print('normal')
          continue

        length = self.__get_edge_length(he1)
        if length<minimum_length or length>maximum_length:
          ok[i] = -1
          # print('len', length, he1)
          continue

        first = self.HE[he1].first
        last = self.HE[he1].last

        x1 = (self.X[first] + self.X[last])*0.5 + normal[0]*h + dx[i]
        y1 = (self.Y[first] + self.Y[last])*0.5 + normal[1]*h + dy[i]

        if vcross(
          self.X[self.HE[he1].first], self.Y[self.HE[he1].first],
          self.X[self.HE[he1].last], self.Y[self.HE[he1].last],
          x1, y1
        )>0:
          ok[i] = -1
          # print('vc')
          continue

        xys[2*i] = x1
        xys[2*i+1] = y1

        ok[i] = self.zonemap.__sphere_is_free(x1, y1, h)
        # print(ok[i] )

      free(normal)

    cdef long res = 0
    cdef long v1
    cdef long he2
    cdef long he3
    cdef long he4
    cdef long f
    cdef long k
    cdef double sx
    cdef double sy
    cdef long bad

    cdef long *remap= <long*>malloc(num*sizeof(long))

    for i in xrange(num):

      if ok[i]<0:
        continue

      he1 = edges[i]

      if self.__is_surface_edge(he1)<0:
        continue

      bad = -1
      for k in xrange(res):
        sx = xys[2*remap[k]] - xys[2*i]
        sy = xys[2*remap[k]+1] - xys[2*i+1]
        if sx*sx+sy*sy<h*h:
          bad = 1
          break

      if bad>-1:
        continue

      v1 = self.__new_vertex(xys[2*i],xys[2*i+1])

      first = self.HE[he1].first
      last = self.HE[he1].last

      self.__set_vertex_intensity(v1, (self.I[self.HE[he1].first] +
                                      self.I[self.HE[he1].last])*0.5)

      he2 = self.__new_edge(last, first)
      self.__set_mutual_twins(he2, he1)
      he3 = self.__new_edge_from_edge(he2, v1)
      he4 = self.__new_edge_from_edge(he3, last)
      self.HE[he4].next = he2

      self.__set_edge_of_vertex(v1,he4)

      f = self.__new_face(he2)

      self.__set_face_of_three_edges(f, he2, he3, he4)
      self.__set_gen_of_three_edges(self.HE[he1].gen+1, he2, he3, he4)

      if merge_ragged_edge>0:
        self.__merge_ragged_edge(he1, he3, he4)

      remap[res] = i
      res += 1

    free(remap)
    free(xys)
    free(ok)

    return res

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cpdef long optimize_position(
    self,
    double attract_scale,
    double reject_scale,
    double triangle_scale,
    double alpha,
    double diminish,
    long scale_intensity
  ):

    return self.__optimize_position(
      attract_scale,
      reject_scale,
      triangle_scale,
      alpha,
      diminish,
      scale_intensity
    )

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cpdef long throw_seed_triangle(
    self,
    long he1,
    double h,
    double dx,
    double dy,
    double rad,
    double rot
  ):
    return self.__throw_seed_triangle(he1, h, dx, dy, rad, rot)

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cpdef long new_triangle_from_surface_edge(
    self,
    long he1,
    double h,
    double dx,
    double dy,
    double minimum_length=0,
    double maximum_length=0,
    long merge_ragged_edge=1
  ):

    return self.__new_triangle_from_surface_edge(
      he1,
      h,
      dx,
      dy,
      minimum_length,
      maximum_length,
      merge_ragged_edge
    )

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  @cython.cdivision(True)
  cpdef long new_triangles_from_surface_edges(
    self,
    long[:] edges,
    long num,
    double h,
    double[:] dx,
    double[:] dy,
    double minimum_length,
    double maximum_length,
    long merge_ragged_edge
  ):
    return self.__new_triangles_from_surface_edges(
      edges,
      num,
      h,
      dx,
      dy,
      minimum_length,
      maximum_length,
      merge_ragged_edge
    )

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cpdef long position_noise(
    self,
    double[:,:] a,
    long scale_intensity
  ):

    cdef long v
    cdef double intensity = 1

    for v in xrange(self.vnum):

      if scale_intensity>0:
        intensity = self.I[v]

      self.X[v] += a[v,0]*intensity
      self.Y[v] += a[v,1]*intensity

    return 1

