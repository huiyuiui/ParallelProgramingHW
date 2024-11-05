NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
HIPCCFLAGS = -std=c++11 -O3 --offload-arch=gfx90a

LDFLAGS = -lm -fopenmp
EXES = hw3-3

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	g++ $(CXXFLAGS) -o $@ $?

hw3-3: hw3-3.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-3-amd: hw3-3.hip
	hipcc $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<
