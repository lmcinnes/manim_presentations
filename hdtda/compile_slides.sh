#! /bin/bash

required_slides=$(grep -v '#' slide_sequence.txt | tr -d ' ' | tr '\n' ' ')
slide_files=$(ls *.py | grep -v "generation")

for slide_file in $slide_files; do
    required_slides_in_file=$(grep -oP '(?<=class )\w+' "$slide_file" | grep -E "$(echo $required_slides | tr ' ' '|')" | tr '\n' ' ')
    if [ -n "$required_slides_in_file" ]; then
        echo "Processing $slide_file for slides: $required_slides_in_file"
        uv run --active manim-slides render -qm --disable_caching $slide_file $required_slides_in_file
    fi
done

uv run --active manim-slides convert --to pptx $required_slides hdtda.pptx
