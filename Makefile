# Find nvcc (NVIDIA CUDA compiler)
NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
		$(error nvcc not found.)
endif

# Compiler flags
CFLAGS = -O3 --use_fast_math
NVCCFLAGS = -lcublas -lcublasLt
# Conditional debug flags
DEBUG_FLAGS = -g -G
PROFILE_FLAGS = -g -lineinfo

# Check for debug mode
ifeq ($(DEBUG),1)
	NVCCFLAGS += $(DEBUG_FLAGS)
endif

ifeq ($(PROFILE),1)
	NVCCFLAGS += $(PROFILE_FLAGS)
endif

train_unet: train_unet.cu
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) $^ -o $@

clean: rm -f train_unet