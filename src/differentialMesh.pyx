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

cdef double TWOPI = 2.*pi


cdef class DifferentialMesh(mesh.Mesh):

  def __init__(self, long nmax, double zonewidth, double nearl, double farl):

    mesh.Mesh.__init__(self, nmax, zonewidth)

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
  cdef long __reject(self, double scale) nogil:
    """
    all vertices will move away from all neighboring (closer than farl)
    vertices
    """

    cdef double farl = self.farl
    cdef double nearl = self.nearl
    cdef long vnum = self.vnum

    cdef long v
    cdef long k
    cdef long neigh

    cdef double x
    cdef double y
    cdef double dx
    cdef double dy
    cdef double nrm
    cdef double s

    cdef double resx = 0.
    cdef double resy = 0.

    cdef long asize = self.zonemap.__get_max_sphere_count()
    cdef long *vertices
    cdef long neighbor_num

    #if True:
    with nogil, parallel(num_threads=4):

      vertices = <long *>malloc(asize*sizeof(long))

      #for v in xrange(vnum):
      for v in prange(vnum, schedule='guided'):

        x = self.X[v]
        y = self.Y[v]

        neighbor_num = self.zonemap.__sphere_vertices(x, y, farl, vertices)

        resx = 0.
        resy = 0.

        for k in range(neighbor_num):

          neigh = vertices[k]

          if neigh == v:

            continue

          dx = x-self.X[neigh]
          dy = y-self.Y[neigh]
          nrm = sqrt(dx*dx+dy*dy)

          if nrm>farl or nrm<=0.0:
            continue

          dx = dx/nrm
          dy = dy/nrm

          s = farl-nrm

          if nrm<nearl:
            s = s*2.

          resx += dx*s
          resy += dy*s

        self.DX[v] += resx
        self.DY[v] += resy

      free(vertices)

    return 1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  @cython.cdivision(True)
  cdef long __edge_vertex_force(self, long he1, long v1, double scale) nogil:

    cdef long henum = self.henum
    cdef double nearl = self.nearl

    cdef long a = self.HE[he1].first
    cdef long b = self.HE[he1].last

    cdef double x = (self.X[b]+self.X[a])*0.5
    cdef double y = (self.Y[b]+self.Y[a])*0.5

    cdef double dx = self.X[v1]-x
    cdef double dy = self.Y[v1]-y
    cdef double nrm = sqrt(dx*dx+dy*dy)

    if nrm<=0:

      return -1

    if vcross(self.X[a], self.Y[a],
      self.X[b], self.Y[b],
      self.X[v1], self.Y[v1])>0:

      ## rotation ok

      if nrm>0.5*sqrt(3.0)*nearl:

        #pass
        self.DX[v1] += -dx/nrm*scale
        self.DY[v1] += -dy/nrm*scale

      else:

        self.DX[v1] += dx/nrm*scale
        self.DY[v1] += dy/nrm*scale

    else:

      ## bad rotation

      self.DX[v1] += -dx/nrm*scale
      self.DY[v1] += -dy/nrm*scale

    return 1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cdef long __triangle_force(self, double scale) nogil:

    cdef long ab
    cdef long bc
    cdef long ca

    for f in xrange(self.fnum):

      ab = self.FHE[f]
      bc = self.HE[ab].next
      ca = self.HE[bc].next

      self.__edge_vertex_force(ab,self.HE[ca].first,scale)
      self.__edge_vertex_force(bc,self.HE[ab].first,scale)
      self.__edge_vertex_force(ca,self.HE[ab].last,scale)

    return 1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  @cython.cdivision(True)
  cdef long __attract(self, double scale) nogil:
    """
    vertices will move towards all connected vertices further away than
    nearl
    """

    cdef long v1
    cdef long v2
    cdef long k

    cdef double nearl = self.nearl

    cdef double dx
    cdef double dy
    cdef double nrm

    cdef double s

    for k in xrange(self.henum):

      v1 = self.HE[k].first
      v2 = self.HE[k].last

      dx = self.X[v2]-self.X[v1]
      dy = self.Y[v2]-self.Y[v1]
      nrm = sqrt(dx*dx+dy*dy)

      if nrm<=0.:
        continue

      if self.HE[k].twin>-1:

        # longernal edge. has two opposing half edges
        # half the force because it is applied twice
        s = scale*0.5/nrm

        if nrm>nearl:

          ## attract
          self.DX[v1] += dx*s
          self.DY[v1] += dy*s

        elif nrm<=nearl:

          ## reject
          self.DX[v1] -= dx*s
          self.DY[v1] -= dy*s

      else:

        # surface edge has one half edge, and they are all rotated the same way
        s = scale/nrm

        if nrm>nearl:

          ## attract
          self.DX[v1] += dx*s
          self.DY[v1] += dy*s
          # and the other vertex
          self.DX[v2] -= dx*s
          self.DY[v2] -= dy*s

        elif nrm<=nearl:

          ## reject

          self.DX[v1] -= dx*s
          self.DY[v1] -= dy*s
          # and the other vertex
          self.DX[v2] += dx*s
          self.DY[v2] += dy*s

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
  cpdef long find_nearby_sources(self):

    return self.__find_nearby_sources()

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cpdef long optimize_position(self, double step):

    cdef long v
    cdef long i

    cdef double reject_scale = 1.0
    cdef double scale = 0.1

    #with nogil:

    double_array_init(self.DX, self.vnum, 0.)
    double_array_init(self.DY, self.vnum, 0.)

    self.__reject(reject_scale)
    self.__attract(scale)
    self.__triangle_force(scale)

    with nogil:
      for v in range(self.vnum):

        self.X[v] += self.DX[v]*step
        self.Y[v] += self.DY[v]*step
        self.zonemap.__update_v(v)

    return 1

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
  cdef long __find_nearby_sources(self) nogil:

    cdef long v
    cdef long n
    cdef double rad = self.source_rad
    cdef long vnum = self.vnum

    cdef long asize = self.source_zonemap.__get_max_sphere_count()
    cdef long *vertices

    cdef long num

    cdef long hits = 0

    vertices = <long *>malloc(asize*sizeof(long))

    for v in xrange(vnum):

      num = self.source_zonemap.__sphere_vertices(self.X[v], self.Y[v], rad, vertices)

      for n in xrange(num):

        self.source_zonemap.__del_vertex(vertices[n])
        self.__set_vertex_intensity(v, 1.0)

        hits += 1

    return hits

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cdef long __smooth_intensity(self, double alpha) nogil:

    cdef long vnum = self.vnum
    cdef long e
    cdef long v
    cdef long v1
    cdef long v2
    cdef double a
    cdef double b

    newintensity = <double *>malloc(vnum*sizeof(double))
    double_array_init(newintensity, vnum, 0.)

    count = <long *>malloc(vnum*sizeof(long))
    long_array_init(count, vnum, 0)

    for e in xrange(self.henum):

      v1 = self.HE[e].first
      v2 = self.HE[e].last

      a = self.I[v1]
      b = self.I[v2]

      newintensity[v1] += ((b-a)*alpha + a*(1.0-alpha))/(1.0-alpha)
      newintensity[v2] += ((a-b)*alpha + b*(1.0-alpha))/(1.0-alpha)

      count[v1] += 1
      count[v2] += 1

    for v in xrange(vnum):

      self.I[v] = newintensity[v]/<double>(count[v])

    free(newintensity)
    free(count)

    return 1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  @cython.cdivision(True)
  cpdef long throw_seed_triangle(
    self,
    long he1,
    double h,
    double dx,
    double dy,
    double rad
  ):

    if self.__is_surface_edge(he1)<0:

      return -1

    if self.__triangle_rotation(self.HE[he1].face)<0:

      return -1

    cdef double *normal = [-1,-1]

    if self.__get_surface_edge_outward_normal(he1, normal)<0:

      return -1

    cdef long first = self.HE[he1].first
    cdef long last = self.HE[he1].last

    cdef double x1 = (self.X[first] + self.X[last])*0.5 + normal[0]*h
    cdef double y1 = (self.Y[first] + self.Y[last])*0.5 + normal[1]*h

    if self.zonemap.__sphere_is_free(x1, y1, h)<0:

      return -1

    return self.__new_faces_in_ngon(x1,y1,rad,3,0.0)

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  @cython.cdivision(True)
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

    cdef double xmid = (self.X[first] + self.X[last])*0.5
    cdef double ymid = (self.Y[first] + self.Y[last])*0.5

    cdef double x1 = xmid + normal[0]*h + dx
    cdef double y1 = ymid + normal[1]*h + dy

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

    cdef long f = self.__new_face(he2)

    self.__set_face_of_three_edges(f, he2, he3, he4)
    self.__set_gen_of_three_edges(self.HE[he1].gen+1, he2, he3, he4)

    if merge_ragged_edge>0:

      self.__merge_ragged_edge(he1, he3, he4)

    return 1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cpdef long smooth_intensity(self, double alpha):

    return self.__smooth_intensity(alpha)

