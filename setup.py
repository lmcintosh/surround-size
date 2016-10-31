from setuptools import setup

setup(name = 'surround',
        version = '0.2',
        description = 'Tools for analyzing intracellular recording data used in surround-size project.',
        author = 'Lane McIntosh',
        author_email = 'lmcintosh@stanford.edu',
        url = 'https://github.com/lmcintosh/surround-size.git',
        long_description = '''
            This package is to streamline Lane's analysis of retinal receptive fields measured
            intra- and extracellularly. Tools to analyze and fit data from David Kastner.
            Also includes functions from efficient coding literature.
            ''',
        classifiers = [
            'Intended Audience :: Science/Research',
            'Operating System :: MacOS :: MacOS X',
            'Topic :: Scientific/Engineering :: Information Analysis'],
        packages = ['surround'],
        package_dir = {'surround': 'surround/'},
        package_data = {'surround': ['data/*.txt']},
        py_modules = ['surround']
        )
