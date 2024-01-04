#!/bin/bash

j=545
while [ $j -lt 1000 ]; do
    echo solving $j
    tsp -i fullRandom_large_${j}.json heuristic greedy | tsp -o fullRandom_large_${j}_tmp.json improve locenum -w 150 -s 75 
    tsp -i fullRandom_large_${j}_tmp.json -o fullRandom_large_${j}_sol.json -c solve bac
    rm fullRandom_large_${j}_tmp.json
    j=$[j+1]
done
