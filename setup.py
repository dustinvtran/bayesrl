try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'bayes-pomdp',
    'description': 'Software for POMDP environments, agents, and their solutions',
    'author': 'Dustin Tran',
    'author_email': 'dtran@g.harvard.edu',
    'version': '0.1',
    'packages': ['bayes-pomdp'],
    'scripts': [],
}

setup(**config)
