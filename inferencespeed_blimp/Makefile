# Makefile for fast_np_count native MicroPython module
# Targets NiclaVision's STM32H747 (Cortex-M7, double-precision FPU)
#
# Usage:
#   make          → builds fast_np_count.mpy
#   make clean    → removes build artifacts

# Path to MicroPython repo (cloned to Desktop)
MPY_DIR = $(HOME)/Desktop/micropython

# Module name and source
MOD = fast_np_count
SRC = fast_np_count.c

# Architecture: Cortex-M7 with double-precision FPU
ARCH = armv7emdp

# Homebrew arm-none-eabi-gcc ships without newlib, so GCC's stdint.h #include_next
# has nothing to find. We provide a minimal ARM Cortex-M7 stub via -idirafter so
# it appears AFTER the GCC internal includes (exactly where include_next looks).
CFLAGS_EXTRA += -idirafter $(CURDIR)/stubs

# Let MicroPython's build system handle all compiler flags and include paths
include $(MPY_DIR)/py/dynruntime.mk
