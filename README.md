# Wide Residual Networks in Keras
A package of Wide Residual Networks for image recognition in Keras.

![Build Status][build-image]

**keras-wrn** is **the** Keras package for Wide Residual Networks. It's fast *and* flexible.

Wide ResNets are faster to train and more accurate than traditional ResNets, even when pre-activation structure is used. 

## Quick example

```python
import keras

import keras_wrn

shape, classes = (32, 32, 3), 10

model = keras_wrn.build_model(shape, classes, 16, 4)

model.compile("adam", "categorical_crossentropy", ["accuracy"])

(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()

y_train = keras.utils.np_utils.to_categorical(training_y)

model.fit(x_train, y_train, epochs=10)
```


## Contribute
Hey there! New ideas are welcome: open/close issues, fork the repo and share your code with a Pull Request.

Clone this project to your computer:

`git clone https://github.com/EricAlcaide/keras-wrn`

By participating in this project, you agree to abide by the thoughtbot [code of conduct](https://thoughtbot.com/open-source-code-of-conduct)

## Meta

* **Author's GitHub Profile**: [Eric Alcaide](https://github.com/EricAlcaide/)
* **Twitter**: [@eric_alcaide](https://twitter.com/eric_alcaide)
* **Email**: ericalcaide1@gmail.com

[build-image]: https://img.shields.io/travis/rust-lang/rust/master.svg "Build Status"