CXX = g++
CXXFLAGS = -std=c++17 -O3 -fopenmp -I ../parlaylib/include
LDFLAGS = -fopenmp

SOURCES = main.cpp methods.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = nbody_sim

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

run: $(EXECUTABLE)
	./$(EXECUTABLE) -N 100000 -d 2  # Default: 100,000 particles in 2D

run3d: $(EXECUTABLE)
	./$(EXECUTABLE) -N 100000 -d 3  # Run with 100,000 particles in 3D

.PHONY: all clean run run3d
