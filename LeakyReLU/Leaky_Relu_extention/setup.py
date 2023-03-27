from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='Leaky_Relu_extention',
      ext_modules=[cpp_extension.CppExtension('myLeakyRelU_cpp', ['myLeakyRelu.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})