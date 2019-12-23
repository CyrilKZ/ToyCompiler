# ToyCompiler

#### A toy compiler translating Python source code into LLVM IRs, based on Python-builtin module *ast* and [*llvmlite*](https://github.com/numba/llvmlite),

&emsp;by tht([CyrilKZ](https://github.com/CyrilKZ)@GitHub) and zhh([fliingelephant](https://github.com/fliingelephant)@GitHub)

- To run this toy compiler, you may install the following package:

    ```bash
    pip3 install llvmlite
    ```
    
- Then compile the given ```source.py```
    ```bash
    python3 ToyCompiler.py
    ```
    
#### As a course project, only few features are supported, not including type inference of function arguments, returning call result of another function in a function, etc.
