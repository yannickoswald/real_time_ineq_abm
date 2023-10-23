# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:36:32 2023

@author: earyo
"""

import pytest


def test_method1():
    x = 5
    y = 10
    assert x == y
    
def test_method2():
    a = 15
    b = 20
    assert a+5 == b
    
test_method1()
test_method2()