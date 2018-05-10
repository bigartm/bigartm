# Copyright 2017, Additive Regularization of Topic Models.

import os
import itertools
import tempfile
import shutil
import pytest
import glob

import artm.wrapper
import artm.wrapper.messages_pb2 as messages
import artm.wrapper.constants as constants
import artm.master_component as mc

def test_func():
    lib = artm.wrapper.LibArtm()

    # The format should be 'MAJOR.MINOR.PATCH'
    versions = lib.version().split('.')
    assert len(versions) == 3
    for version in versions:
        assert version.isdigit()
