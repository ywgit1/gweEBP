#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 24/02/2020
# @Author  : Fangliang Bai
# @File    : utils.py
# @Software: PyCharm
# @Description:
"""


def fix_parameters(module):
    for param in module.parameters():
        param.requires_grad = False
