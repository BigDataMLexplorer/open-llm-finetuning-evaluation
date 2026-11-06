import numpy as np
from peft_lab.eval import expected_calibration_error

def test_ece_bounds():
    p=np.array([[.9,.1],[.2,.8]]);y=np.array([0,1]);e=expected_calibration_error(p,y)
    assert 0 <= e <= 1
