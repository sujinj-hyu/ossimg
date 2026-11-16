import pytest
from PIL import Image

from ossimg import grayscale, mean_brightness


# Normal input: the example in the docstring gives the documented result
def test_mean_brightness_of_gradient():
    result = mean_brightness("examples/test.png")
    assert result == 127.5


# Boundary input: the darkest possible image
def test_mean_brightness_of_black_image(tmp_path):
    black = tmp_path / "black.png"
    Image.new("L", (10, 10), 0).save(black)
    assert mean_brightness(black) == 0.0


# Normal input, checked through the file the function writes
def test_grayscale_writes_gray_image(tmp_path):
    out = tmp_path / "gray.png"
    grayscale("examples/test.png", out)
    assert Image.open(out).mode == "L"


# Wrong input: the function must refuse to overwrite its own input
def test_grayscale_refuses_to_overwrite():
    with pytest.raises(ValueError):
        grayscale("examples/test.png", "examples/test.png")