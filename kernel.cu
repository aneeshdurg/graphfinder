#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>

#define MAX_GRAPHS (1 << 20)
#define THREADS_PER_BLOCK 256
#define MAX_NODES 16
#define MAX_EDGES (MAX_NODES * MAX_NODES)

#define CuAlloc(sz)                                                            \
  ({                                                                           \
    void *tmp = NULL;                                                          \
    auto err = cudaMalloc(&tmp, sz);                                           \
    if (tmp == NULL) {                                                         \
      fprintf(stderr, "Failed to allocate device vector!\n");                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
    tmp;                                                                       \
  })

#define min(a, b)                                                              \
  ({                                                                           \
    auto tmpa = (a);                                                           \
    auto tmpb = (b);                                                           \
    tmpa < tmpb ? tmpa : tmpb;                                                 \
  })

#define max(a, b)                                                              \
  ({                                                                           \
    auto tmpa = (a);                                                           \
    auto tmpb = (b);                                                           \
    tmpa > tmpb ? tmpa : tmpb;                                                 \
  })

__host__ __device__ size_t choose(size_t n, size_t r) {
  size_t a = min(n - r, r);
  size_t b = max(n - r, r);

  size_t num = 1;
  for (size_t i = 0; i < a; i++) {
    num *= (n - i);
  }

  size_t denom = 1;
  for (size_t i = 1; i <= a; i++) {
    denom *= i;
  }

  return num / denom;
}

/**
https://stackoverflow.com/a/57076790

def combination_at_idx(idx, elems, r):
    if len(elems) == r:
        # We are looking for r elements in a list of size r - thus, we need
        # each element.
        return elems

    if len(elems) == 0 or len(elems) < r:
        return []

    combinations = choose(len(elems), r)    # total number of combinations
    remains = choose(len(elems) - 1, r)     # combinations after selection

    offset = combinations - remains

    if idx >= offset:       # combination does not start with first element
        return combination_at_idx(idx - offset, elems[1:], r)

    # We now know the first element of the combination, but *not* yet the next
    # r - 1 elements. These need to be computed as well, again recursively.
    return [elems[0]] + combination_at_idx(idx, elems[1:], r - 1)
*/

__device__ void combination_at_idx(size_t idx, size_t *elems_out,
                                   size_t first_elem, size_t n, size_t r) {
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
  // count how many nodes a DFS rooted at `root` reaches
  size_t frontier[MAX_EDGES];
  size_t frontier_len = root;
  frontier[frontier_len++] = 0;

  size_t visited[MAX_NODES];
  size_t visited_len = 0;

  while (frontier_len > 0) {
    size_t nid = frontier[--frontier_len];

    if (contains(visited, visited_len, nid)) {
      continue;
    }
    visited[visited_len++] = nid;

    for (size_t i = 0; i < n_edges; i++) {
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

  return visited_len;
}

/**
 * Returns true if the graph described by the src/dst edge pairs is fully
 * connected, false otherwise
 */
__device__ bool is_connected(size_t n_nodes, size_t n_edges, size_t *srcs,
                             size_t *dsts) {
  return n_connected(0, n_nodes, n_edges, srcs, dsts) == n_nodes;
}

__global__ void project_tm_to_graph(size_t offset, uint32_t *out,
                                    size_t *elist_src, size_t *elist_dst,
                                    size_t n_graphs, size_t n_nodes,
                                    size_t n_edges) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i > n_graphs) {
    // This index is invalid, and doesn't correspond to a valid graph
    return;
  }

  size_t n_total_edges = n_nodes * (n_nodes - 1) / 2;

  // Get the i'th combination of selecting `n_edges` edges from all possible
  // edges.
  size_t g[MAX_EDGES];
  combination_at_idx(offset + i, g, 0, n_total_edges, n_edges);

  // Map the edge id to a src/dst pair
  size_t src[MAX_EDGES];
  size_t dst[MAX_EDGES];
  for (size_t eid = 0; eid < n_edges; eid++) {
    src[eid] = elist_src[g[eid]];
    dst[eid] = elist_dst[g[eid]];
  }

  if (!is_connected(n_nodes, n_edges, src, dst)) {
    out[i] = 0;
  } else {
    out[i] = 1;
  }
}

__host__ std::vector<std::string> split(std::string s, std::string delimiter) {
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  std::string token;
  std::vector<std::string> res;

  while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }

  res.push_back(s.substr(pos_start));
  return res;
}

int main(int argc, char **argv) {
  cudaError_t err;

  assert(argc >= 4);

  size_t n_nodes = atoi(argv[1]);
  size_t n_edges = atoi(argv[2]);
  size_t n_total_edges = n_nodes * (n_nodes - 1) / 2;
  assert(n_edges <= n_total_edges);

  char const *filename = argv[3];

  printf("n_nodes = %zu\n", n_nodes);
  printf("n_edges = %zu\n", n_edges);

  // Read the input matrix to be projected onto each graph
  std::vector<std::vector<size_t>> traffic_matrix(n_nodes);
  for (size_t i = 0; i < n_nodes; i++) {
    traffic_matrix[i] = std::vector<size_t>(n_nodes);
  }

  std::ifstream ifile(filename);
  std::string str;
  size_t src = 0;
  while (std::getline(ifile, str)) {
    auto parts = split(str, " ");
    for (size_t dst = 0; dst < parts.size(); dst++) {
      traffic_matrix[src][dst] += atoi(parts[dst].c_str());
    }
    src++;
  }

  thrust::device_vector<size_t> tm(n_nodes * n_nodes);
  for (size_t i = 0; i < n_nodes; i++) {
    for (size_t j = 0; j < n_nodes; j++) {
      tm[i * n_nodes + j] = traffic_matrix[i][j];
    }
  }

  printf("n_graphs = %zu C %zu\n", n_total_edges, n_edges);
  size_t n_graphs = choose(n_total_edges, n_edges);
  printf("n_graphs = %zu\n", n_graphs);

  uint32_t *output = (uint32_t *)CuAlloc(sizeof(uint32_t) * MAX_GRAPHS);
  printf("Allocated output vector\n");

  std::vector<size_t> srcs;
  std::vector<size_t> dsts;
  for (size_t i = 0; i < n_nodes; i++) {
    for (size_t j = (i + 1); j < n_nodes; j++) {
      srcs.push_back(i);
      dsts.push_back(j);
    }
  }
  auto dev_src_list = (size_t *)CuAlloc(sizeof(size_t) * srcs.size());
  printf("Allocated edge src list\n");
  auto dev_dst_list = (size_t *)CuAlloc(sizeof(size_t) * dsts.size());
  printf("Allocated edge dst list\n");

  err = cudaMemcpy(dev_src_list, srcs.data(), sizeof(size_t) * srcs.size(),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(dev_dst_list, dsts.data(), sizeof(size_t) * dsts.size(),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    exit(EXIT_FAILURE);
  }

  size_t acc = 0;

  for (size_t offset = 0; offset < n_graphs; offset += MAX_GRAPHS) {
    size_t n_graphs_iter = min(MAX_GRAPHS, n_graphs - offset);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (n_graphs_iter + threadsPerBlock - 1) / threadsPerBlock;
    // printf("CUDA kernel launch with %d blocks of %d threads\n",
    // blocksPerGrid,
    //        threadsPerBlock);
    project_tm_to_graph<<<blocksPerGrid, threadsPerBlock>>>(
        offset, output, dev_src_list, dev_dst_list, n_graphs, n_nodes, n_edges);

    // Wrap raw ptr with a device_ptr so that we can call thrust::reduce below
    thrust::device_ptr<uint32_t> dev_ptr = thrust::device_pointer_cast(output);

    // Get the number of connected graphs by summing up the output
    // TODO eventually the kernel will return the cost of projecting the traffic
    // matrix onto the graph being evaluated - we'll do a custom reduction op
    // that does a min but also tells us which graph produced the minimum cost
    auto x = thrust::reduce(dev_ptr, dev_ptr + n_graphs_iter, 0,
                            thrust::plus<uint32_t>());

    // Copy results to host
    // size_t *host_output = (size_t *)malloc(sizeof(size_t) * n_graphs);
    // err = cudaMemcpy(host_output, output, sizeof(size_t) * n_graphs,
    //                  cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess) {
    //   fprintf(stderr,
    //           "Failed to copy output from device to host (error code %s)!\n",
    //           cudaGetErrorString(err));
    //   exit(EXIT_FAILURE);
    // }

    // for (size_t i = 0; i < n_graphs; i++) {
    //   printf(" graph[%zu] is connected? %zu\n ", i, host_output[i]);
    // }

    acc += (size_t)x;
    printf("Remaining %zu\n", n_graphs - offset);
  }
  cudaFree(dev_src_list);
  cudaFree(dev_dst_list);
  cudaFree(output);
  printf("x=%zu\n", acc);
}
