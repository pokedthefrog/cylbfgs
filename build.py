#!/usr/bin/env python
from distutils.ccompiler import get_default_compiler

import numpy
from setuptools import Extension
from setuptools.command.build_ext import build_ext


# noinspection PyUnresolvedReferences,PyUnboundLocalVariable
def build(setup_kwargs):
    try:
        from Cython.Build import cythonize
        use_cython = True
    except ImportError:
        use_cython = False

    class CustomExtBuilder(build_ext):
        def finalize_options(self):
            build_ext.finalize_options(self)

            compiler = self.compiler
            compiler = get_default_compiler() if compiler is None \
                else compiler

            if compiler == 'msvc':
                include_dirs.append('compat/win32/')

    include_dirs = [
        'liblbfgs/include/',
        'liblbfgs/lib/',
        numpy.get_include(),
    ]

    if use_cython:
        ext_modules = cythonize([
            Extension('cylbfgs._lowlevel',
                      sources=[
                          'cylbfgs/_lowlevel.pyx',
                          'liblbfgs/lib/lbfgs.c',
                      ],
                      include_dirs=include_dirs),
        ])
    else:
        ext_modules = [
            Extension('cylbfgs._lowlevel',
                      sources=[
                          'cylbfgs/_lowlevel.c',
                          'liblbfgs/lib/lbfgs.c',
                      ],
                      include_dirs=include_dirs),
        ]

    setup_kwargs.update({
        'ext_modules': ext_modules,
        'cmdclass': {'build_ext': CustomExtBuilder, },
    })


if __name__ == '__main__':
    pass
