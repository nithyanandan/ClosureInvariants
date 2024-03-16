import setuptools, re, glob, os, sys
from setuptools import setup, find_packages

# githash = 'unknown'
# if os.path.isdir(os.path.dirname(os.path.abspath(__file__))+'/.git'):
#     try:
#         gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout = PIPE)
#         githash = gitproc.communicate()[0]
#         if gitproc.returncode != 0:
#             print("unable to run git, assuming githash to be unknown")
#             githash = 'unknown'
#     except EnvironmentError:
#         print("unable to run git, assuming githash to be unknown")
# githash = githash.decode('utf-8').replace('\n', '')

# with open(os.path.dirname(os.path.abspath(__file__))+'/ClosureInvariants/githash.txt', 'w+') as githash_file:
#     githash_file.write(githash)

with open('./ClosureInvariants/__init__.py', 'r') as metafile:
    metafile_contents = metafile.read()
    metadata = dict(re.findall("__([a-z]+)__\s*=\s*'([^']+)'", metafile_contents))
    
pkg_data={'astroutils': ['*.txt', 'examples/cosmotile/*.yaml',
                         'examples/image_cutout/*.yaml',
                         'examples/catalogops/*.yaml',
                         'examples/codes/lightcone_operations/*.py',
                         'examples/codes/lightcone_operations/*.yaml']}

install_req_list=['astroutils']

setup_req_list = ['astroutils']


setup(name='ClosureInvariants',
    # version=metadata['version'],
    # description=metadata['description'],
    # long_description=open("README.rst").read(),
    # url=metadata['url'],
    # author=metadata['author'],
    # author_email=metadata['authoremail'],
    # maintainer=metadata['maintainer'],
    # maintainer_email=metadata['maintaineremail'],
    license='MIT',
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.7+',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Astronomy',
                 'Topic :: Utilities'],
    packages=find_packages(),
    package_data = pkg_data,
    include_package_data=True,
    scripts=glob.glob('scripts/*.py'),
    install_requires=install_req_list,
    setup_requires=setup_req_list,
    tests_require=['pytest'],
    zip_safe=False)
