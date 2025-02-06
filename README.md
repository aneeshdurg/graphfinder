# graphfinder

Given:
+ `n` the number of routers
+ `r` the number of links
+ `tm` a traffic matrix where `tm[i][j]` indicates the number of bytes being
  sent from router `i` to router `j`

`graphfinder` will find the topology connecting `n` routers with `r` links
that minimizes the total number of bytes sent for the workload `tm`.

## Status

Currently `graphfinder` aims to be a GPU accelerated brute force search so
that I compare other strategies/heuristic approximations to the optimal
solution.
