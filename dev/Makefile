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


# Default rule for our CUDA files
%.o: %.cu
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) -c -D LINKING $< -o $@

%: %.cu
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) common.o $< -o $@

PHONY: all clean

# Build all targets
BINARIES = groupnorm linear avgpool upsample silu conv2d_k1 conv2d_k3 broadcast common add attention concat_channel mse timestep_embedding
OBJECTS = $(addsuffix .o, $(BINARIES))
UNET_OBJECTS = $(OBJECTS) resblock.o attention_block.o
TARGETS = $(OBJECTS) resblock
all: $(TARGETS)

groupnorm.o: groupnorm.cu
linear.o: linear.cu
avgpool.o: avgpool.cu
upsample.o: upsample.cu
silu.o: silu.cu
conv2d_k1.o: conv2d_k1.cu
conv2d_k3.o: conv2d_k3.cu
broadcast.o: broadcast.cu
common.o: common.cu
add.o: add.cu
attention.o: attention.cu

groupnorm: groupnorm.cu
linear: linear.cu
avgpool: avgpool.cu
upsample: upsample.cu
silu: silu.cu
conv2d_k1: conv2d_k1.cu
conv2d_k3: conv2d_k3.cu
broadcast: broadcast.cu
mse: mse.cu

resblock: resblock.cu $(OBJECTS)
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) $^ -o $@

attention_block: attention_block.cu $(OBJECTS)
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) $^ -o $@

unet: unet.cu $(UNET_OBJECTS)
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) $^ -o $@

unet_test: unet_test.cu $(UNET_OBJECTS)
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) $^ -o $@

unet_test_long: unet_test_long.cu $(UNET_OBJECTS)
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) $^ -o $@

clean:
	rm -f $(TARGETS) $(OBJECTS) $(BINARIES) resblock attention_block unet unet_test unet_test_long