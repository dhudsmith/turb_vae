from setuptools import find_packages, setup

setup(
    name='turb_vae',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=2.1.0',
    ],
    author='D. Hudson Smith',
    author_email='dane2@clemson.edu',
    description='Variational Autoencoders for optical turbulence phase screens',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dhudsmith/turb_vae',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)