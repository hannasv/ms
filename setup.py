# -*- coding: utf-8 -*-

from setuptools import setup

#with open('README.rst') as file:
#readme = file.read()

#with open('VERSION.md') as f:
#    version = f.readline()
#    print(version)
#    f.close()


setup(
    name        =   'sciclouds',
    version     =   '0.1',
    author      =   'Hanna Svennevik',
    author_email=   'svennevikh@gmail.com',
    url         =   'https://github.com/hannasv/MS.git',
    license     =   'GPLv3',
    package_dir =   {'sclouds'      :   'sclouds'},
    packages    =   ['sclouds',
                     'sclouds.io',
                     'sclouds.plot',
                     'sclouds.stats',
                     'sclouds.ml'],
    #find_packages(exclude=['contrib', 'docs', 'tests*']),
    #include_package_data sci-clouds True,
    package_data=   {'sclouds'    :   ['data/*']},

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',

        # Pick your license as you wish (should match 'license' above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.,
        'Programming Language :: Python :: 3.6'
    ],

    install_requires    =   [],
    dependency_links    =   [],
    description = ('Python tools for my master thesis, where i do cloud cover prediction.'),
    #long_description = readme,
    #entry_points = {'console_scripts' : ['pyaerocom=pyaerocom.scripts.main:cli']},
    zip_safe = False
)
