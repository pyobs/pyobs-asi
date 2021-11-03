from setuptools import setup

setup(
    name='pyobs-asi',
    version='0.14',
    description='pyobs component for ASI cameras',
    author='Tim-Oliver Husser',
    author_email='thusser@uni-goettingen.de',
    packages=['pyobs_asi'],
    install_requires=[
        'zwoasi',
        'numpy',
        'astropy'
    ]
)
