# Differential Mesh

![ani](/img/ani2.gif?raw=true "animation")

Differential Mesh is an extension of the Hyphae
(https://github.com/inconvergent/hyphae) algorithm. I started working on it
with the intention of mimicking the growth of (certain types of) lichen. The
results, to me, are equal parts crystal-like and biological.

![ani](/img/ani.gif?raw=true "animation")

![img](/img/img.jpg?raw=true "image")

The algorithm appends triangles to the outside edges of the structure while it
tries to keep themes from intersecting with itself. If the triangles are made
small enough it will look like some sort of biological process.

The grid can be built in one of two modes. The basic one is the static mode.
Once a triangle has been positioned, it will remain there forever. In the other
mode the vertices will move throughout the process: They will avoid collisions
with nearby vertices, whilst at the same time remaining close by their
neighbours.

![img](/img/img3.jpg?raw=true "image")

## Prerequisites

In order for this code to run you must first download and install these two
repositories:

*    `iutils`: https://github.com/inconvergent/iutils
*    `zonemap`: https://github.com/inconvergent/zonemap

## Other Dependencies

The code also depends on:

*    `numpy`
*    `scipy`
*    `cython`
*    `python-cairo` (do not install with pip, this generally does not work)

## Running it on Linux (Ubuntu)

To install the libraries locally, run `./install`. I have only tested this code
in Ubuntu 14.04 LTS, but my guess is that it should work on most other
platforms platforms as well.  However i know that the scripted install in
`./install` will not work in Windows

## Similar code

If you find this alorithm insteresting you might also want to check out:
https://github.com/inconvergent/differential-line.

-----------
http://inconvergent.net

