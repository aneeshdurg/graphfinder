.PHONY: clean
.PHONY: all

all: graphfinder graphfinder.ptx

graphfinder: graphfinder.cu
	nvcc -o graphfinder graphfinder.cu

graphfinder.ptx: graphfinder.cu
	nvcc -O0 -o graphfinder.ptx graphfinder.cu -ptx -src-in-ptx

clean:
	rm -vf graphfinder graphfinder.ptx
