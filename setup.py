from setuptools import setup, find_packages

setup(
    name='pysweat',
    version='0.1.dev1',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        'pymongo>=3',
        'pandas>=0.16'
    ],
    url='https://github.com/jsamoocha/pysweat',
    license='Apache',
    author='Jonatan Samoocha',
    author_email='',
    description=''
)
