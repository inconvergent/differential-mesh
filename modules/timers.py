# -*- coding: utf-8 -*-

from time import time

class named_sub_timers(object):

  def __init__(self, name=None):

    from collections import defaultdict

    self.name = name
    self.times = defaultdict(float)
    self.now = time()
    self.total = 0.

  def start(self):

    self.now = time()

  def t(self,n):

    t = time()
    diff = t-self.now

    self.times[n] += diff
    self.total += diff

    self.now = t

  def p(self):

    print('{:s}'.format('' if not self.name else self.name))

    for n,t in self.times.iteritems():

      print('{:s}\t{:0.6f}'.format(n,t))

    print('total\t{:0.6f}\n'.format(self.total))

