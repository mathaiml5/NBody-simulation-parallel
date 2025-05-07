#!/bin/bash

# Create the results directory if it doesn't exist
RESULTS_DIR="results"
mkdir -p $RESULTS_DIR

# Function to run simulation and continue even if it fails
run_simulation() {
    N=$1
    DIM=$2
    ACCURACY=$3  # Whether to enable accuracy calculation (0 or 1)
    
    echo "Running simulation for N=$N, dimension=$DIM, accuracy=$ACCURACY"
    
    # Run the simulation with proper arguments, continue on error
    ./nbody_sim -N $N -d $DIM -a $ACCURACY || {
        echo "Simulation failed for N=$N, dimension=$DIM"
        echo "Moving to next simulation..."
    }
    
    echo "Finished simulation for N=$N, dimension=$DIM"
    echo "------------------------------------------"
}

# List of N values to run
declare -a N_VALUES=(1000 10000 100000 200000 500000 1000000 2000000 5000000)
declare -a N_EXP=("1e3" "1e4" "1e5" "2e5" "5e5" "1e6" "2e6" "5e6")

# Ensure the executable is built
echo "Building the executable..."
make clean && make || {
    echo "Build failed. Exiting."
    exit 1
}
echo "Build completed."
echo "------------------------------------------"

# Run 2D simulations with accuracy off
echo "Starting 2D simulations (without accuracy)..."
for N in "${N_VALUES[@]}"; do
    run_simulation $N 2 0
done

# Run 3D simulations with accuracy off
echo "Starting 3D simulations (without accuracy)..."
for N in "${N_VALUES[@]}"; do
    run_simulation $N 3 0
done

# Run 2D simulations with smaller datasets and accuracy on
echo "Starting 2D simulations with accuracy calculation..."
for N in "${N_VALUES[@]:0:4}"; do  # Only run for first 4 sizes
    run_simulation $N 2 1
done

# Run 3D simulations with smaller datasets and accuracy on
echo "Starting 3D simulations with accuracy calculation..."
for N in "${N_VALUES[@]:0:4}"; do  # Only run for first 4 sizes
    run_simulation $N 3 1
done

echo "All simulations completed for timestamp: $DATE_STAMP"
echo "Results are available in the $RESULTS_DIR directory"
