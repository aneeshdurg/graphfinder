#include <cstddef>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#define __host__
#define __device__

__host__ __device__ size_t factorial(size_t x) {
  if (x <= 1)
    return 1;

  return x * factorial(x - 1);
}

__host__ __device__ size_t choose(size_t n, size_t r) {
  return factorial(n) / (factorial(n - r) * factorial(r));
}

void combination_at_idx(size_t idx, size_t *elems_out, size_t first_elem,
                        size_t n, size_t r) {
  if (first_elem == (n - r)) {
    for (size_t i = 0; i < r; i++) {
      elems_out[i] = first_elem + i;
    }
  }

  if (first_elem >= (n - r)) {
    return;
  }

  size_t combos = choose(n - first_elem, r);
  size_t remains = choose(n - first_elem - 1, r);

  size_t offset = combos - remains;
  if (idx >= offset) {
    return combination_at_idx(idx - offset, elems_out, first_elem + 1, n, r);
  }

  elems_out[0] = first_elem;
  return combination_at_idx(idx, elems_out + 1, first_elem + 1, n, r - 1);
}

__device__ bool contains(size_t *bag, size_t len, size_t elem) {
  for (size_t i = 0; i < len; i++) {
    if (bag[i] == elem) {
      return true;
    }
  }
  return false;
}

__device__ size_t n_connected(size_t root, size_t n_nodes, size_t n_edges,
                              size_t *srcs, size_t *dsts) {
  // check that a DFS starting at 0 reaches all nodes

  size_t ctr = 0;
  size_t frontier[64];
  size_t frontier_len = root;
  frontier[frontier_len++] = 0;

  size_t visited[64];
  size_t visited_len = 0;
  size_t src_cat = 0;
  size_t dst_cat = 0;

  while (frontier_len > 0) {
    size_t nid = frontier[--frontier_len];

    printf("    visiting %zu\n", nid);

    if (contains(visited, visited_len, nid)) {
      printf("      already visited %zu\n", nid);
      continue;
    }
    visited[visited_len++] = nid;

    for (size_t i = 0; i < n_edges; i++) {
      printf("      checking edge %zu-%zu\n", srcs[i], dsts[i]);
      if (srcs[i] == nid) {
        if (!contains(visited, visited_len, dsts[i])) {
          frontier[frontier_len++] = dsts[i];
        }
      } else if (dsts[i] == nid) {
        if (!contains(visited, visited_len, srcs[i])) {
          frontier[frontier_len++] = srcs[i];
        }
      }
    }
  }

  {
    size_t src_cat = 0;
    size_t dst_cat = 0;
    for (size_t i = 0; i < n_edges; i++) {
      src_cat *= 10;
      src_cat += srcs[i];
      dst_cat *= 10;
      dst_cat += dsts[i];
    }
    ctr += src_cat * 1000 + dst_cat;
    ctr += 1000000 * (srcs == dsts ? 1 : 2);
  }

  return ctr * 100 + visited_len;
}

/**
 * Returns true if the graph described by the src/dst edge pairs is fully
 * connected, false otherwise
 */
__device__ bool is_connected(size_t n_nodes, size_t n_edges, size_t *srcs,
                             size_t *dsts) {
  return n_connected(0, n_nodes, n_edges, srcs, dsts) == n_nodes;
}

int main() {
  size_t n = 3;
  size_t r = 2;
  size_t elems[r];
  size_t src[r];
  size_t dst[r];

  std::vector<size_t> srcs;
  std::vector<size_t> dsts;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = (i + 1); j < n; j++) {
      srcs.push_back(i);
      dsts.push_back(j);
    }
  }

  size_t n_combos = choose(n, r);
  printf("n_combos=%zu\n", n_combos);

  for (size_t idx = 0; idx < n_combos; idx++) {
    printf("combination %zu\n", idx);
    combination_at_idx(idx, elems, 0, n, r);

    for (size_t i = 0; i < r; i++) {
      printf("  elems[%zu]=%zu\n", i, elems[i]);
      src[i] = srcs[elems[i]];
      dst[i] = dsts[elems[i]];
    }
    size_t n_connect = n_connected(0, n, r, src, dst);
    printf("  n_connect = %zu\n", n_connect);
    break;
  }
}
