#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 23:30:04 2021

@author: loc
"""
#%% class method overwriding
class Base(object):
    
    def __init__(self, x=1, y=1):
        self.x = x
        self.y=y
        self(1.5, 1.5)
    def func_add(self,):
        self.add()
        
    def add(self, x=2, y=2):
        print(x+y)
        
class Child(Base):
    def __init__(self, x=1, y=1):
        super().__init__()
        # self.x = x
        # self.y = y
    def add(self, x=5, y=5):
        print(x + y)
        
        
base_class = Base()
child_class = Child()
child_class.func_add()

