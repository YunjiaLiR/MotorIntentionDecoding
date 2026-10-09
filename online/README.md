# Online decoder development (experimental)

This folder contains *model-training/export experiments* and MATLAB scripts used in developing or testing a motor-imagery decoder. The presence of a script does **not** establish that it was successfully deployed online.

- `model_training/`: Python FBCSP/hierarchical training and export variants.
- `matlab/`: MATLAB data preparation, model training/evaluation and acquisition helpers.

The separate `online.zip` test archive supplied during the project was not accessible for byte extraction during the GitHub import. Consequently, **its contents have not been reviewed or imported**. No validated, fully working real-time system is claimed.

Some scripts expect fixed channel mappings, run indices, folder layouts and external recordings. Verify these before attempting to reproduce them.
