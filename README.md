# benchmark

```bash
python setup.py build_ext --inplace
uv run -m benchmake.sgemm
```

# compile cute kernel

cutlass version: v4.2.1

```bash
mkdir build & cd build
cmake ..
make
```

# debug triton kernel

```bash
.venv/bin/activate
TRITON_INTERPRET=1 python -m pdb csrc/triton/matmul.py
```
