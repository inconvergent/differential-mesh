# -*- coding: utf-8 -*-
# cython: profile=True

from __future__ import division

cimport mesh
from libc.stdlib cimport malloc, free
from zonemap cimport Zonemap

cimport numpy as np


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

  cdef long __optimize_position(
    self,
    double attract_scale,
    double reject_scale,
    double triangle_scale,
    double alpha,
    double diminish,
    long scale_intensity
  ) nogil

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
  )

  cdef long __edge_vertex_force(
    self,
    long v1,
    double *diffx,
    double *diffy,
    long he1,
    double scale
  ) nogil

  cdef long __triangle(
    self,
    long v1,
    double *diffx,
    double *diffy,
    double scale,
    long *vertices,
    long num
  ) nogil

  cdef long __merge_ragged_edge(self, long he1, long he3, long he4) nogil

  cdef long __find_nearby_sources(self, long hit_limit)

  cdef long __smooth_intensity(
    self,
    long v1,
    double alpha,
    double *old,
    double *new,
    long *vertices,
    long num
  ) nogil

  ## EXTERNAL

  cpdef long initialize_sources(self, list sources, double source_rad)

  cpdef long find_nearby_sources(self, long hit_limit=*)

  cpdef long optimize_position(
    self,
    double attract_scale,
    double reject_scale,
    double triangle_scale,
    double alpha,
    double diminish,
    long scale_intensity
  )

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
  )

  cpdef long position_noise(
    self,
    double[:,:] a,
    long scale_intensity
  )

