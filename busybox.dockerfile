# Use the BusyBox base image
FROM busybox

# Set the command to execute when a container is started from this image
CMD ["echo", "Howdy cowboy"]
