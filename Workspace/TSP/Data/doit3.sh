#!/bin/bash

j=0
while [ $j -lt 1000 ]; do
    echo solving $j
    tsp -i fullRandom_large_${j}.json -o fullRandom_large_${j}_greedy.json heuristic greedy
    j=$[j+1]
done
