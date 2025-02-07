/**
 * Given a number of routers N, a number of links R, and a NxN traffic matrix TM
 * where TM[i][j] is the number of bytes sent from router i to router j,
 * Find G, the topology with N nodes and R edges such that the cost of sending
 * the data in TM is minimized, where the cost is defined as the total number
 * of bytes sent by all routers when traffic is routed by a shortest-path
 * policy.
 */

#include <fstream>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>

#define MAX_GRAPHS (1 << 28)
#define THREADS_PER_BLOCK 256

// To allow static allocation of data structures on the stack of GPU threads, we
// set a constant MAX_NODES that this program accepts here.
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

// Tuple of graph id and cost, this can't just be a std::pair because std::pair
// isn't defined with __device__.
struct Pair {
  size_t graph;
  size_t cost;

  __host__ __device__ Pair(size_t graph, size_t cost)
      : graph(graph), cost(cost) {}

  __host__ __device__ Pair() : graph(0), cost(0) {}

  __host__ __device__ Pair operator=(Pair other) {
    this->graph = other.graph;
    this->cost = other.cost;
    return *this;
  }
};

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

__host__ __device__ void combination_at_idx(size_t idx, size_t *elems_out,
                                            size_t first_elem, size_t n,
                                            size_t r) {
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

__global__ void project_tm_to_graph(size_t offset, Pair *out,
                                    size_t *traffix_matrix, size_t *elist_src,
                                    size_t *elist_dst, size_t n_graphs,
                                    size_t n_nodes, size_t n_edges) {
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
    out[i] = Pair(i, 0);
  } else {

    // Floyd Warshall Algorithm
    size_t min_dists[MAX_EDGES];

    for (size_t src_nid = 0; src_nid < n_nodes; src_nid++) {
      for (size_t dst_nid = 0; dst_nid < n_nodes; dst_nid++) {
        min_dists[src_nid * n_nodes + dst_nid] = n_edges + 1;
      }
    }

    for (size_t eid = 0; eid < n_edges; eid++) {
      size_t src_nid = src[eid];
      size_t dst_nid = dst[eid];

      min_dists[src_nid * n_nodes + dst_nid] = 1;
      min_dists[dst_nid * n_nodes + src_nid] = 1;
    }

    for (size_t i_node = 0; i_node < n_nodes; i_node++) {
      for (size_t src_nid = 0; src_nid < n_nodes; src_nid++) {
        for (size_t dst_nid = (src_nid + 1); dst_nid < n_nodes; dst_nid++) {
          min_dists[src_nid * n_nodes + dst_nid] =
              min(min_dists[src_nid * n_nodes + dst_nid],
                  min_dists[src_nid * n_nodes + i_node] +
                      min_dists[i_node * n_nodes + dst_nid]);
        }
      }
    }

    size_t cost = 0;
    for (size_t src_nid = 0; src_nid < n_nodes; src_nid++) {
      for (size_t dst_nid = src_nid + 1; dst_nid < n_nodes; dst_nid++) {
        auto n_hops = min_dists[src_nid * n_nodes + dst_nid];
        cost += n_hops * traffix_matrix[src_nid * n_nodes + dst_nid];
        cost += n_hops * traffix_matrix[dst_nid * n_nodes + src_nid];
      }
    }

    out[i] = Pair(offset + i, cost);
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

struct mingraph {
  using T = struct Pair;
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef T result_type;

  __thrust_exec_check_disable__ __host__ __device__ struct Pair
  operator()(const T &lhs, const T &rhs) const {
    if (lhs.cost == 0) {
      return rhs;
    }

    if (rhs.cost == 0) {
      return lhs;
    }
    return (lhs.cost < rhs.cost) ? lhs : rhs;
  }
};

int main(int argc, char **argv) {
  cudaError_t err;

  assert(argc >= 4);

  size_t n_nodes = atoi(argv[1]);
  size_t n_edges = atoi(argv[2]);
  size_t n_total_edges = n_nodes * (n_nodes - 1) / 2;
  // Connected graphs need at least N - 1 edges
  assert(n_edges > (n_nodes - 1));
  // Check that n_edges isn't more than the number of possible edges because we
  // aren't considering multigraphs
  assert(n_edges <= n_total_edges);
  assert(n_nodes <= MAX_NODES);

  char const *filename = argv[3];

  printf("n_nodes = %zu\n", n_nodes);
  printf("n_edges = %zu\n", n_edges);

  // Read the input matrix to be projected onto each graph
  std::vector<size_t> traffic_matrix(n_nodes * n_nodes);

  std::ifstream ifile(filename);
  std::string str;
  size_t src = 0;
  while (std::getline(ifile, str)) {
    auto parts = split(str, " ");
    for (size_t dst = 0; dst < parts.size(); dst++) {
      traffic_matrix[src * n_nodes + dst] +=
          strtoul(parts[dst].c_str(), NULL, 10);
    }
    src++;
  }

  // Copy traffic matrix to GPU
  size_t *tm = (size_t *)CuAlloc(sizeof(size_t) * n_nodes * n_nodes);
  err = cudaMemcpy(tm, traffic_matrix.data(),
                   sizeof(size_t) * traffic_matrix.size(),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    exit(EXIT_FAILURE);
  }

  // Count the total number of graphs with n_nodes nodes and n_edges edges
  printf("n_graphs = %zu C %zu\n", n_total_edges, n_edges);
  size_t n_graphs = choose(n_total_edges, n_edges);
  printf("n_graphs = %zu\n", n_graphs);

  // allocate output buffer (pair of graph-id, cost of projection) - for large
  // values of n_nodes we can't allocate a large enough buffer, so instead we
  // choose a fixed size output and process one chunk of graphs at a time.
  Pair *output = (Pair *)CuAlloc(sizeof(Pair) * MAX_GRAPHS);
  printf("Allocated output vector\n");

  // Create a mapping from edge ids in [0, n_total_edges) to (src, dst)
  // i.e. the edge with id eid's source and dest is given by srcs[eid],
  // dsts[eid] respectively.
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

  // Comparator for finding the minimum element in output
  auto min_fn = mingraph();

  // Min element so far
  Pair acc = Pair();

  for (size_t offset = 0; offset < n_graphs; offset += MAX_GRAPHS) {
    // Number of graphs processed by this iteration
    size_t n_graphs_iter = min(MAX_GRAPHS, n_graphs - offset);

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (n_graphs_iter + threadsPerBlock - 1) / threadsPerBlock;
    project_tm_to_graph<<<blocksPerGrid, threadsPerBlock>>>(
        offset, output, tm, dev_src_list, dev_dst_list, n_graphs, n_nodes,
        n_edges);

    // Wrap raw ptr with a device_ptr so that we can call thrust::reduce below
    thrust::device_ptr<Pair> dev_ptr = thrust::device_pointer_cast(output);

    // Get the minimum graph-id and the associated cost for the current chunk
    auto curr_min =
        thrust::reduce(dev_ptr, dev_ptr + n_graphs_iter, Pair(0, 0), min_fn);

    // Copy results to host (for debugging)
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

    printf("Remaining %zu\n", n_graphs - offset);
    // On the CPU take the minimum value for this chunk and compare it with the
    // known minimum (if no minimum is known - i.e. cost=0 - then this will
    // always pick `curr_min`)
    acc = min_fn(acc, curr_min);
  }
  cudaFree(dev_src_list);
  cudaFree(dev_dst_list);
  cudaFree(output);

  printf("min graph[%zu] has cost %zub\n", acc.graph, acc.cost);
  // Use the graph-id to find the set of edges that formed the minimum cost
  // solution
  printf("  graph[%zu] = [", acc.graph);
  size_t g[MAX_EDGES];
  combination_at_idx(acc.graph, g, 0, n_total_edges, n_edges);
  for (size_t eid = 0; eid < n_edges; eid++) {
    printf(" (%zu, %zu)", srcs[g[eid]], dsts[g[eid]]);
  }
  printf(" ]\n");
}
