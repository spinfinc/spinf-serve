"""Quantization: the schemes' kernels (i8gemm, w4gemm) and the per-revision tables (tables/: fp8s activation scales, i8 outlier
channels), made by calibrate."""

import os

TABLES_DIR = os.path.join(os.path.dirname(__file__), "tables")
