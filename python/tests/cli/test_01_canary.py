from subprocess import call
import helper

def test_func():
    bigartm = helper.find_bigartm_binary()
    call([bigartm, "--help"])
