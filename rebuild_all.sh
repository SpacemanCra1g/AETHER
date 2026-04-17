#!/usr/bin/zsh

for d in build-*; do 
    cmake --build "$d" -j &
done 
wait