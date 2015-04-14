try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'bayesrl',
    'description': 'Reinforcement learning using Bayesian approaches',
    'author': 'Dustin Tran',
    'author_email': 'dtran@g.harvard.edu',
    'version': '0.1',
    'packages': ['bayesrl'],
    'scripts': [],
}

setup(**config)
