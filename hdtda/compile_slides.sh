#! /bin/bash

required_slides=$(ggrep -v '#' slide_sequence.txt | tr -d ' ' | tr '\n' ' ')
slide_files=$(ls *.py | ggrep -v "generation")

for slide_file in $slide_files; do
    required_slides_in_file=$(ggrep -oP '(?<=class )\w+' "$slide_file" | ggrep -E "$(echo $required_slides | tr ' ' '|')" | tr '\n' ' ')
    if [ -n "$required_slides_in_file" ]; then
        echo "Processing $slide_file for slides: $required_slides_in_file"
        uv run manim-slides render --disable_caching $slide_file $required_slides_in_file
    fi
done

uv run manim-slides convert --to pptx $required_slides hdtda_1080p.pptx
