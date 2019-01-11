# ERT Simple

We provide C++ code in order to replicate our ERT baseline based on http://www.csc.kth.se/~vahidk/face_ert.html

#### Requisites
- faces_framework https://github.com/bobetocalo/faces_framework

#### Installation
This repository must be located inside the following directory:
```
faces_framework
    └── alignment 
        └── ert_simple
```
You need to have a C++ compiler (supporting C++11):
```
> mkdir release
> cd release
> cmake ..
> make -j$(nproc)
> cd ..
```
#### Usage
Use the --database option to load the proper trained model.
```
> ./release/face_alignment_ert_simple_test --database 300w_public --measure pupils
```
