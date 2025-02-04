# graphfinder

Given:
+ `n` the number of routers
+ `r` the number of links
+ `tm` a traffic matrix where `tm[i][j]` indicates the number of bytes being
  sent from router `i` to router `j`

`nvgraphfinder` will find the topology connecting `n` routers with `r` links
that minimizes the total number of bytes sent for the workload `tm`.

## Status

Currently `nvgraphfinder` aims to be a GPU accelerated brute force search so
that I compare other strategies/heuristic approximations to the optimal
solution.

+`nvgraphfinder` can currently enumerate the number of connected graphs with `n`
nodes and `r` edges.
  + The next step is to apply the traffic matrix to connected graphs and find the
graph with minimum total cost
+ For reasonable values of `n` (e.g. 16) the current implementation is still
  really slow - can it be done faster?
