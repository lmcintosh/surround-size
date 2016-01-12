from setuptools import setup

from setuptools import setup

setup(name = 'surround-size',
        version = '0.1',
        description = 'Tools for analyzing intracellular recording data used in surround-size project.',
        author = 'Lane McIntosh',
        author_email = 'lmcintosh@stanford.edu',
        url = 'https://github.com/baccuslab/surround-size.git',
        long_description = '''
            This package is to streamline Lane's analysis of retinal receptive fields measured
            intra- and extracellularly. Tools to analyze and fit data from David Kastner.
            Also includes functions from efficient coding literature.
            ''',
        classifiers = [
            'Intended Audience :: Science/Research',
            'Operating System :: MacOS :: MacOS X',
            'Topic :: Scientific/Engineering :: Information Analysis'],
        packages = ['surround-size'],
        package_dir = {'surround-size': ''},
        py_modules = ['atick_redlich_functions', 'get_davids_data', 'image_processing_functions',
            'lnl_model', 'lnl_model_functions']
        )
