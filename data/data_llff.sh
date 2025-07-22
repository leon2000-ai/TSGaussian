video_in=$1
cd $(dirname "$video_in")

mkdir -p input
# Extract the frame in the video folder
ffmpeg -i $video_in -qscale:v 1 -qmin 1 -vf fps=2 input/%04d.jpg
cd ..
pwd
# Then move all the generated images to the input folder
python convert.py -s $(dirname "$video_in") #If not resizing, ImageMagick is not needed
# LLFF bound
python ./LLFF/imgs2poses.py $(dirname "$video_in")
