#!/bin/bash

for i in {0..4};
do
  for j in {0..1};
  do
    python3 CNN_angle.py $i $j
  done
done
