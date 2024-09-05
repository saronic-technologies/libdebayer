# -*- mode: bash-ts -*-
# shellcheck shell=bash

# Base URL of the Kodak image set
base_url="https://r0k.us/graphics/kodak/kodak/"

# Loop through images 1 to 24
for i in $(seq 1 24); do
  # Format the image number with leading zero if needed
  image_number=$(printf "%02d" "$i")

  # Construct the image URL
  image_url="${base_url}kodim${image_number}.png"

  # Construct the output filename
  filename="kodim${image_number}.png"

  # Use wget to download the image
  # Check if the download was successful
  if wget "$image_url" -O "$filename"; then
    echo "Downloaded: $filename"
  else
    echo "Error downloading: $filename"
  fi
done
