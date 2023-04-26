from setuptools import setup

setup(name='pygamry',
      version='0.1',
      description='A Python package for customized data acquisition with Gamry instruments',
      # long_description=long_description,
      # long_description_content_type="text/markdown",
      # url='https://github.com/jdhuang-csm/bayes-drt2',
      author='Jake Huang',
      author_email='jdhuang@mines.edu',
      license='BSD 3-clause',
      packages=['pygamry'],
      install_requires=[
            'numpy',
            'scipy',
            'pandas',
            'matplotlib',
            'comtypes'
      ],
      include_package_data=True
      )
