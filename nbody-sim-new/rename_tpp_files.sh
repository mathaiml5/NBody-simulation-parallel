#!/bin/bash
# Script to rename all .tpp files to .cpp and update references

# Find all .tpp files and rename them
for file in $(find . -name "*.tpp"); do
    new_file="${file%.tpp}.cpp"
    mv "$file" "$new_file"
    echo "Renamed $file to $new_file"
done

# Update all includes and references in source files
for file in $(find . -name "*.h" -o -name "*.cpp"); do
    # Replace #include statements
    sed -i 's/\.tpp"/.cpp"/g' "$file"
    sed -i 's/\.tpp>/.cpp>/g' "$file"
    
    # Update any other references
    sed -i 's/\.tpp/.cpp/g' "$file"
    
    echo "Updated references in $file"
done

echo "Renaming complete. Please check for any missed references."
