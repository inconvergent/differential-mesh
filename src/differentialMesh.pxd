# -*- coding: utf-8 -*-
# cython: profile=True

from __future__ import division

cimport mesh
from libc.stdlib cimport malloc, free
from zonemap cimport Zonemap


cdef class DifferentialMesh(mesh.Mesh):

  cdef double nearl

  cdef double farl

  cdef long num_sources

  cdef double source_rad

  cdef double *DX

  cdef double *DY

  cdef double *SX

  cdef double *SY

  cdef Zonemap source_zonemap

  ## FUNCTIONS

  cdef long __reject(
    self,
    long v1,
    long *vertices,
    long num,
    double scale,
    double *sx,
    double *sy
  ) nogil

  cdef long __attract(
    self,
    long v1,
    long *connected,
    long num,
    double scale,
    double *dxdx,
    double *dydy
  ) nogil

  cdef long __optimize_position(self, double step) nogil

  cdef long __throw_seed_triangle(
    self,
    long he1,
    double h,
    double dx,
    double dy,
    double rad,
    double rot
  ) nogil

  cdef long __new_triangle_from_surface_edge(
    self,
    long he1,
    double h,
    double dx,
    double dy,
    double minimum_length,
    double maximum_length,
    long merge_ragged_edge
  ) nogil

  cdef long __edge_vertex_force(
    self,
    long v1,
    long *opposite,
    long num,
    double scale,
    double *dxdx,
    double *dydy
  ) nogil

  cdef long __merge_ragged_edge(self, long he1, long he3, long he4) nogil

  cdef long __find_nearby_sources(self) nogil

  cdef long __smooth_intensity(self, double alpha) nogil

  ## EXTERNAL

  cpdef long initialize_sources(self, list sources, double source_rad)

  cpdef long find_nearby_sources(self)

  cpdef long optimize_position(self, double step)

  cpdef long throw_seed_triangle(
    self,
    long he1,
    double h,
    double dx,
    double dy,
    double rad,
    double rot
  )

  cpdef long new_triangle_from_surface_edge(
    self,
    long he1,
    double h,
    double dx,
    double dy,
    double minimum_length=*,
    double maximum_length=*,
    long merge_ragged_edge=*
  )

  cpdef long smooth_intensity(self, double alpha)


