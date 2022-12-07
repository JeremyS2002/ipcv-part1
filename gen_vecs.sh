#!/bin/bash
for n in {00..09}
do
opencv_createsamples -img Streets/$n.jpg -vec vecs/$n.vec -w 20 -h 20 -num 100 -maxidev 80 -maxxangle 0.1 -maxyangle 0.8 -maxzangle 0.1
done
