"""Shared nvcc options for runtime and toolkit-only builds."""

FLAGS = ("-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-allow-unsupported-compiler")
