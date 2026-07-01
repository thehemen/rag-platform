# Minetest Assistant

This project implements the detection of Minetest MineClone2 mobs.

A visual description of the project implementation is available on [Medium](https://medium.com/@thehemen/minetest-assistant-mob-detection-to-make-game-simple-0090c323f7d1).

## Usage

To use this application, just run:

```sh
python3 main.py
```

## Installation

To use it, you need to install the required pip-packages:

```sh
pip install PyYaml mss pynput ultralytics
```

Among packages, the "opencv-python" package is installed. It may cause the conflict between Ultralytics and PyQt5 packages.
To avoid it, just remove this package:

```sh
pip uninstall opencv-python
```

Then, install the necessary deb-packages:

```sh
apt-get install python3-pyqt5 pyqt5-dev-tools qttools5-dev-tools
apt-get install python3-opencv
```

Good luck!
