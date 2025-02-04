.PHONY: clean
.PHONY: all

all: kernel kernel.ptx

kernel: kernel.cu
	nvcc -o kernel kernel.cu

kernel.ptx: kernel.cu
	nvcc -O0 -o kernel.ptx kernel.cu -ptx -src-in-ptx

clean:
	rm -vf kernel kernel.ptx
