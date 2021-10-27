from setuptools import setup, find_packages

setup(
    name='rl_motap',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'tensorflow',
        'gym',
        'pyglet',
        'numpy',
        'tqdm'
    ]
)