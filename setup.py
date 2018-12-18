from setuptools import find_packages, setup

setup(
    name='lcp_physics',
    version='0.1.0',
    description='A differentiable LCP physics engine in PyTorch.',
    author='Filipe de Avila Belbute-Peres',
    author_email='filiped@cs.cmu.edu',
    platforms=['any'],
    url='https://github.com/locuslab/lcp-physics',
    packages=find_packages(exclude=['demos', 'videos']),
    install_requires=['py3ode', 'pygame']
)
