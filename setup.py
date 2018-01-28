from setuptools import setup

setup(name='keras-wrn',
      version='0.2',
      description='The Keras package for Wide Residual Networks',
      url='https://github.com/EricAlcaide/keras-wrn',
      author='Eric Alcaide',
      author_email='ericalcaide1@gmail.com',
      license='MIT',
      packages=['keras_wrn'],
      install_requires=["keras"],
      zip_safe=False)