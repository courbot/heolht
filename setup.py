from setuptools import setup

setup(
    name='heolht',
    version='2.0',
    packages=['heolht'],
    package_dir={'heolht': 'lib'},
    zip_safe=False,
    install_requires=['numpy', 'scipy'],
)
