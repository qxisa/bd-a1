#!/bin/bash

CONTAINER_NAME="finalContainer"
OUTPUT_DIRECTORY="./service-result"

echo "Creating output directory: $OUTPUT_DIRECTORY"
mkdir -p "$OUTPUT_DIRECTORY"

echo "Copying files from container to local directory..."

docker cp "$CONTAINER_NAME:/home/doc-bd-a1/res_dpre.csv" "$OUTPUT_DIRECTORY/res_dpre.csv"
docker cp "$CONTAINER_NAME:/home/doc-bd-a1/eda-in-1.txt" "$OUTPUT_DIRECTORY/eda-in-1.txt"
docker cp "$CONTAINER_NAME:/home/doc-bd-a1/eda-in-2.txt" "$OUTPUT_DIRECTORY/eda-in-2.txt"
docker cp "$CONTAINER_NAME:/home/doc-bd-a1/eda-in-3.txt" "$OUTPUT_DIRECTORY/eda-in-3.txt"
docker cp "$CONTAINER_NAME:/home/doc-bd-a1/vis.png" "$OUTPUT_DIRECTORY/vis.png"
docker cp "$CONTAINER_NAME:/home/doc-bd-a1/k.txt" "$OUTPUT_DIRECTORY/k.txt"

echo "Stopping container: $CONTAINER_NAME"
docker stop "$CONTAINER_NAME"

echo "Output files copied and container stopped."
