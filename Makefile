C = g++
CFLAGS = -std=c++17 -Wall -O2 -fopenmp
OBJ_FILES = utils.o methods.o main.o

all: nbody

nbody: $(OBJ_FILES)
	$(C) $(CFLAGS) -o $@ $^

%.o: %.cpp
	$(C) $(CFLAGS) -c $< -o $@

clean:
	rm -f nbody $(OBJ_FILES)