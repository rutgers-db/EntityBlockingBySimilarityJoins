# End-to-End Entity Matching System

## Introduction
The Entity Matching (EM) problem involves the identification of tuple pairs from one or two sets of instances that correspond to the same real-world entities. A typical EM solution comprises two main steps: **blocking** and **matching**. The blocking step aims to eliminate tuple pairs that are evidently non-matched, while the matching step evaluates the survived pairs to reach a final decision.

Modern EM solutions often prioritize enhancing the accuracy of the matching step, and negelect the recall of the blocking step. Additionally, the popular packages which support various blocking techniques are usually built on Python, which, while ensuring portability, often results in limitations concerning blocker's scalability.

Therefore, we porpose to design an entity matching system that emphasizes the recall during the blocking step. Meanwhile, we maintain a focus on the matcher, to enhance the recall in this phase, we propose to integrate a **value matcher** in the system. Our proposed solution is designed to be scalable, as the blocker is assembled by state-of-art similarity join algorithms that are written in C++ and enhanced by parallelization, while also maintaining portability through the provision of public APIs written in Python.

## The System Design

Our system encompasses five stpes: 
1. Sample the input set(s) to train a random forest matcher.
2. Extracting blocking rules from the matcher to assemble the rule-based blocker (RBB) and apply the desired blocker on input set(s).
3. Extract features and calculate scores of the blocking results.
4. (Random forest) Predict each tuple pair as matched/non-matched in blocking results.
5. (Value matcher) Indentify the interchangeable values from the matching results and repeat step 3 and 4 by considering interchangeable values during calculating features scores.

The step 1-3 are implemented in C++ and step 4-5 are implemented in Python. All public APIs are written in Python.

The project layout is:
1. ```bin``` contains the executable files of sample (step 1), block (step 2) and feature extraction (step 3).
2. ```shared_lib``` contains the compiled dynamic library of the three steps as in ```bin```.
3. ```cpp``` contains the C++ source code.
4. ```scripts``` contains the bash wrapper for running the binary executable files in ```bin```.
5. ```simjoin_entitymatching``` contains the public APIs for our system.
6. ```examples``` contains the scripts as examples ro run our system.
7. ```test``` contains the unit tests and experiment scripts, you should not use them.

### Prerequisite

#### Python
Required Python packages are listed in ```requirements.txt```. For ```py_entitymatching```, please refer to their [documents](https://anhaidgroup.github.io/py_entitymatching/v0.4.0/user_manual/guides.html) for more details. Meanwhile, we have several minor modifications on this package, which are inlcuded in the ```patch/py_em.patch```. (Note: the patch file has not been tested, you may want to refer to ```docs/modifications.md``` and modify the package manually)

#### C++
The default C++ compiler is ```g++``` and C++ version is ```11```, you may use any other compilers as long as they support ```OpenMP```. Additionally, if you would like to use ```parquet``` format for input and output, you should install ```arrow``` package and de-comment the marco ```ARROW_INSTALLED``` in ```cpp/common/config.h``` as well as compile settings in all ```CMakeLists.txt```. But the ```parquet``` io reamins untested at this stage.

### How to build?
```
bash build.sh
```
This commands will invoke the root ```CMakeLists.txt``` and compiles all the parts (sample, block and feature) to generate the binary files as well as dynamic libraries. The compile log is written in ```build/compile.log```.

### How to run the system?

See in the ```exmples``` folder and the corresponding ```README.md```.

### How to port parts of out system in your EM solution?

Refer to the ```light-weight``` branch for more information. Coming soon...

### Other documents
1. The illustrations of running binary files in ```docs/binary.md```
2. The illustrations of running dynamic library files in ```docs/lib.md```
3. The illustrations of similarity join algorithms in ```docs/simjoin.md```
4. The experiments are in ```docs/exp.md```
5. Please ignore the ```docs/developer_notes.md```

## Acknowledgement
1. The python part of our EM system is built on the package [py_entitymatching](https://github.com/anhaidgroup/py_entitymatching).

2. The implementation of similarity join algorithms are adapted from [rutgers-db/SIGMOD2022-Programming-Contest-Public](https://github.com/rutgers-db/SIGMOD2022-Programming-Contest-Public), [rutgers-db/RedPajama_Analysis](https://github.com/rutgers-db/RedPajama_Analysis) and the original implementation of the corresponding papers by Prof. Deng.

## Platform
Linux.

## Contact
Author: [Yunqi Li](https://ericlyunqi.github.io/), email: ylilo@connect.ust.hk, HKUST. Advised by Prof. [Dong Deng](https://people.cs.rutgers.edu/~dd903/), Rutgers University.

## TODO
Refer to ```docs/TODO.md```.