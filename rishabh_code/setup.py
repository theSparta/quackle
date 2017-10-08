from distutils.core import setup, Extension
from Cython.Build import cythonize
import glob
from os.path import abspath

setup(ext_modules = cythonize(Extension(
       "move",                             # the extension name
       sources= ["move.pyx", "move_create.cpp"],               # the Cython source and
       extra_objects = ["../test/obj/release/trademarkedboards.o"] + glob.glob(abspath("../obj/release/*")), # additional C++ source files
       include_dirs=[abspath("../"), "/usr/include/qt4/QtCore", "/usr/include/qt4"],
       library_dirs=['../quackleio/lib/debug', '../lib/release', '../quackleio/lib/release'],
       libraries=['quackleio', 'quackle', 'QtCore'],
       language="c++",                        # generate and compile C++ code
       extra_compile_args=["-std=c++11", "-fPIC"],
       extra_link_args=["-std=c++11"],
  )))