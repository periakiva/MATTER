#!/usr/bin/env python

"""The setup script."""
from os.path import exists

from setuptools import setup, find_packages


def parse_version(fpath):
    """
    Statically parse the version number from a python file
    """
    import ast
    if not exists(fpath):
        raise ValueError('fpath={!r} does not exist'.format(fpath))
    with open(fpath, 'r') as file_:
        sourcecode = file_.read()
    pt = ast.parse(sourcecode)
    class VersionVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if getattr(target, 'id', None) == '__version__':
                    self.version = node.value.s
    visitor = VersionVisitor()
    visitor.visit(pt)
    return visitor.version

VERSION = parse_version('MATTER/__init__.py')

# with open('README.rst') as readme_file:
#     readme = readme_file.read()

requirements = []

setup(
    author="Peri Akiva",
    author_email='peri.akiva@rutgers.edu',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="",
    entry_points={
        'console_scripts': [
            'watch_hello_world=watch.tools.hello_world:main',
        ],
    },
    install_requires=requirements,
    long_description_content_type='text/x-rst',
    # long_description=readme,
    include_package_data=True,
    name='MATTER',
    packages=find_packages(include=['MATTER', 'MATTER.*']),
    url='https://github.com/periakiva/MATTER.git',
    version=VERSION,
    zip_safe=False,
)
