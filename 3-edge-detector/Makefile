# environment
SM := 35
INPUT_IMG=input_image.jpg
OUTPUT_IMG=output_image.jpg
GCC := g++
NVCC := nvcc

# Remove function
RM = rm -f
 
# Specify opencv Installation
#opencvLocation = /usr/local/opencv
opencvLIB= -L/shared/apps/opencv-3.0.0-beta/INSTALL/lib
opencvINC= -I/shared/apps/opencv-3.0.0-beta/INSTALL/include

# Compiler flags:
# -g    debugging information
# -Wall turns on most compiler warnings
GENCODE_FLAGS := -gencode arch=compute_$(SM),code=sm_$(SM)
LIB_FLAGS := -lcudadevrt -lcudart

NVCCFLAGS :=
GccFLAGS = -fopenmp -O3 

debug: GccFLAGS += -DDEBUG -g -Wall
debug: NVCCFLAGS += -g -G
debug: all

# The build target executable:
TARGET = sobel

all: build

build: $(TARGET)

$(TARGET): src/dlink.o src/main.o src/$(TARGET).o
	$(NVCC) $(NVCCFLAGS) $(opencvLIB) $(opencvINC) $^ -o $@ $(GENCODE_FLAGS) -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_photo -lopencv_video #-link `pkg-config opencv --cflags --libs`

src/dlink.o: src/$(TARGET).o 
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -dlink

src/main.o: src/main.cpp
	$(GCC) $(GccFLAGS) $(opencvLIB) $(opencvINC) -c $< -o $@
	
src/$(TARGET).o: src/$(TARGET).cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS) 

run: all
	fswebcam -r 1920x1080 --jpeg 85 -D 1 ./$(INPUT_IMG) && sleep 4 && ./$(TARGET) ./$(INPUT_IMG) ./$(OUTPUT_IMG) && feh ./$(INPUT_IMG) && feh ./$(OUTPUT_IMG)

run2: all
	./$(TARGET) ~/Pictures/Webcam/1.jpg ./output_image.jpg && feh ./output_image.jpg & feh ~/Pictures/Webcam/1.jpg


clean:
	$(RM) $(TARGET) src/*.o *.o *.tar* *.core* *out*.jpg *input*.jpg
