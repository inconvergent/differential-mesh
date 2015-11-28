# -*- coding: utf-8 -*-
# cython: profile=True

cimport numpy as np
from zonemap cimport Zonemap

cdef struct s_HE:
  long id # half edge id
  long gen # generation of edge
  long first # vertex 1
  long last # vertex 2
  long twin # twin edge
  long next # next edge id
  long face # adjacent face id

ctypedef s_HE sHE


cdef class Mesh:

  cdef long nmax
  cdef long vnum
  cdef long henum
  cdef long fnum

  cdef long nz
  cdef double zonewidth

  cdef long procs

  ## ARRAYS

  cdef double *X
  cdef double *Y
  cdef double *I
  cdef double *DI
  cdef sHE *HE
  cdef long *VHE
  cdef long *FHE

  cdef Zonemap zonemap

  ## FUNCTIONS

  ## longERNAL

  cdef long __valid_new_vertex(self, double x, double y) nogil

  cdef long __new_vertex(self, double x, double y) nogil

  cdef long __new_edge(self, long first, long last) nogil

  cdef long __new_edge_from_edge(self, long he1, long last) nogil

  cdef long __new_face(self, long he1) nogil

  cdef void __set_face_of_three_edges(self, long face, long he1, long he2, long he3) nogil

  cdef void __set_gen_of_three_edges(self, long gen, long he1, long he2, long he3) nogil

  cdef void __set_edge_of_face(self, long face, long he1) nogil

  cdef void __set_mutual_twins(self, long he1, long he2) nogil

  cdef void __set_edge_of_vertex(self, long ve1, long he1) nogil

  cdef long __get_surface_edge_outward_normal(self, long he1, double *nn) nogil

  cdef long __get_surface_edge_outward_vector(self, long he1, double *nn) nogil

  cdef long __is_surface_edge(self, long t1) nogil

  cdef long __next_surface(self, long he1, long direction) nogil

  cdef long __edge_duplicate_test(self, long he1, long a, long b) nogil

  cdef long __flip_edge(self, long he1, double limit) nogil

  cdef long __split_edge(self, long he1) nogil

  cdef long __set_next_of_triangle(self, long he1, long he2, long he3) nogil

  cdef long __split_internal_edge(self, long he1) nogil

  cdef long __split_surface_edge(self, long he1) nogil

  cdef long __split_all_longest_triangle_edges(self, double limit) nogil

  cdef double __get_edge_length(self, long he1) nogil

  cdef long __edge_integrity(self, long he1) nogil

  cdef long __triangle_integrity(self, long face) nogil

  cdef long __triangle_rotation(self, long face) nogil

  cdef long __safe_vertex_positions(self, double limit) nogil

  cdef double __get_edge_intensity(self, long he1) nogil

  cdef void __set_vertex_intensity(self, long v1, double i) nogil

  cpdef long set_vertices_intensity(self, long[:], double i)

  cdef void __set_edge_intensity(self, long he1, double i) nogil

  cdef void __add_edge_intensity(self, long he1, double i) nogil

  cdef long __new_faces_in_ngon(self, double x1, double y1, double rad, long num, double angle) nogil

  cdef long __new_face_in_triangle(self, double x1, double y1, double rad, double angle) nogil

  cdef long __get_connected_vertices(self, long v1, long *tmp) nogil

  cdef long __get_opposite_edges(self, long v1, long *tmp) nogil

  ## EXTERNAL

  cpdef long edge_integrity(self, long he1)

  cpdef long triangle_integrity(self, long face)

  cpdef long triangle_rotation(self, long face)

  cpdef long safe_vertex_positions(self, double limit)

  cpdef long new_face(self, double x1, double y1, double x2, double y2, double x3, double y3)

  cpdef long new_faces_in_ngon(self, double x1, double y1, double rad, long num, double angle)

  cpdef long next_surface(self, long he1, long direction)

  cpdef long flip_edge(self, long he1, double limit)

  cpdef long split_edge(self, long he1)

  cpdef long optimize_edges(self, double split_limit, double flip_limit)

  ## GET DATA

  cpdef long np_get_triangles_coordinates(self, double[:,:] a)

  cpdef long np_get_edges_coordinates(self, double[:,:] a)

  cpdef long np_get_internal_edges_coordinates(self, double[:,:] a)

  cpdef long np_get_triangles_gen(self, long[:] a)

  cpdef long np_get_edges_gen(self, long[:] a)

  cpdef long np_get_edges_gen_diff(self, long[:] a)

  cpdef double get_edge_intensity(self, long he1)

  cpdef long set_vertex_intensity(self, long v1, double i)

  cpdef long set_edge_intensity(self, long he1, double i)

  cpdef long add_edge_intensity(self, long he1, double i)

  cpdef double get_triangle_intensity(self, long f1)

  cpdef list get_triangle_edges(self, long f1)

  ## INFO

  cpdef long is_surface_edge(self, long t1)

  cpdef double get_edge_length(self, long he1)

  cpdef dict get_edge_dict(self, long he1)

  cpdef list get_triangle_dicts(self, long f)

  cpdef long get_vnum(self)

  cpdef long get_henum(self)

  cpdef long get_fnum(self)

