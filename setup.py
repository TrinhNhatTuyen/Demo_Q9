from distutils.core import setup
from Cython.Build import cythonize
directives = {'linetrace': False, 'language_level': 3}
setup(
    ext_modules = cythonize("Run_Fire.py")
    # ext_modules = cythonize("Run_Pose.py")
)
