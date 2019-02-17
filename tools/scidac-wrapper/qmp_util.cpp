#include <cstdlib>

#include <qmp.h>
#include <qmp_util.h>

// Original code from QUDA library

Topology *default_topo = NULL;

static bool comms_initialized = false;
static int rank_order         = 0;

int gridsize_from_cmdline[4] = {1,1,1,1};

void comm_set_default_topology(Topology *topo)
{
  default_topo = topo;
}

Topology *comm_default_topology(void)
{
  if (!default_topo) {
    printf("Default topology has not been declared");
		exit(-1);
  }
  return default_topo;
}

/**
 * Utility function for indexing into Topology::ranks[]
 *
 * @param ndim  Number of grid dimensions in the network topology
 * @param dims  Array of grid dimensions
 * @param x     Node coordinates
 * @return      Linearized index cooresponding to the node coordinates
 */
static inline int index(int ndim, const int *dims, const int *x)
{
  int idx = x[0];
  for (int i = 1; i < ndim; i++) {
    idx = dims[i]*idx + x[i];
  }
  return idx;
}


static inline bool advance_coords(int ndim, const int *dims, int *x)
{
  bool valid = false;
  for (int i = ndim-1; i >= 0; i--) {
    if (x[i] < dims[i]-1) {
      x[i]++;
      valid = true;
      break;
    } else {
      x[i] = 0;
    }
  }
  return valid;
}

Topology *comm_create_topology(int ndim, const int *dims, CommsMap rank_from_coords, void *map_data)
{
  if (ndim > QMP_MAX_DIM) {
    printf("ndim exceeds QMP_MAX_DIM"); exit(-1);
  }

  Topology *topo = (Topology *) malloc(sizeof(Topology));

  topo->ndim = ndim;

  int nodes = 1;
  for (int i=0; i<ndim; i++) {
    topo->dims[i] = dims[i];
    nodes *= dims[i];
  }

  topo->ranks = (int *) malloc(nodes*sizeof(int));
  topo->coords = (int (*)[QMP_MAX_DIM]) malloc(nodes*sizeof(int[QMP_MAX_DIM]));

  int x[QMP_MAX_DIM];
  for (int i = 0; i < QMP_MAX_DIM; i++) x[i] = 0;

  do {
    int rank = rank_from_coords(x, map_data);
    topo->ranks[index(ndim, dims, x)] = rank;
    for (int i=0; i<ndim; i++) {
      topo->coords[rank][i] = x[i];
    }
  } while (advance_coords(ndim, dims, x));

  int my_rank = comm_rank();
  topo->my_rank = my_rank;
  for (int i = 0; i < ndim; i++) {
    topo->my_coords[i] = topo->coords[my_rank][i];
  }

  // initialize the random number generator with a rank-dependent seed
  //rand_seed = 17*my_rank + 137;

  return topo;
}

int comm_rank(void)
{
  return QMP_get_node_number();
}


int comm_size(void)
{
  return QMP_get_number_of_nodes();
}

void comm_init(int ndim, const int *dims, CommsMap rank_from_coords, void *map_data)
{
  if ( QMP_is_initialized() != QMP_TRUE ) {
    printf("QMP has not been initialized"); exit(-1);
  }

  int grid_size = 1;
  for (int i = 0; i < ndim; i++) {
    grid_size *= dims[i];
  }
  if (grid_size != QMP_get_number_of_nodes()) {
    printf("Communication grid size declared via initCommsGridQuda() does not match, total number of QMP nodes (%d != %d)", grid_size, QMP_get_number_of_nodes());
  }

  Topology *topo = comm_create_topology(ndim, dims, rank_from_coords, map_data);
  comm_set_default_topology(topo);
}

const int *comm_dims(const Topology *topo)
{
  return topo->dims;
}

int comm_dim(int dim)
{
  Topology *topo = comm_default_topology();
  return comm_dims(topo)[dim];
}


/**
 * For MPI, the default node mapping is lexicographical with t varying fastest.
 */
static int lex_rank_from_coords(const int *coords, void *fdata)
{
  auto *md = static_cast<LexMapData *>(fdata);

  int rank = coords[0];
  for (int i = 1; i < md->ndim; i++) {
    rank = md->dims[i] * rank + coords[i];
  }
  return rank;
}

/**
 * For QMP, we use the existing logical topology if already declared.
 */
static int qmp_rank_from_coords(const int *coords, void *fdata)
{
  return QMP_get_node_number_from(coords);
}

void initCommsGridQMP(int nDim, const int *dims, CommsMap func, void *fdata)
{
  if (nDim != 4) {
    printf("Number of communication grid dimensions must be 4"); exit(-1);
  }

  LexMapData map_data;
  if (!func) {

    if (QMP_logical_topology_is_declared()) {
      if (QMP_get_logical_number_of_dimensions() != 4) {
        printf("QMP logical topology must have 4 dimensions");
				exit(-1);
      }
      for (int i=0; i<nDim; i++) {
        int qdim = QMP_get_logical_dimensions()[i];
        if(qdim != dims[i]) {
          printf("QMP logical dims[%d]=%d does not match dims[%d]=%d argument", i, qdim, i, dims[i]);
					exit(-1);
        }
      }
      fdata = nullptr;
      func = qmp_rank_from_coords;
    } else {
      printf("QMP logical topology is undeclared; using default lexicographical ordering");

      map_data.ndim = nDim;
      for (int i=0; i<nDim; i++) {
        map_data.dims[i] = dims[i];
      }
      fdata = (void *) &map_data;
      func = lex_rank_from_coords;
    }

  }

  comm_init(nDim, dims, func, fdata);
  comms_initialized = true;

	return;
}

static int lex_rank_from_coords_t(const int *coords, void *fdata)
{
  int rank = coords[0];
  for (int i = 1; i < 4; i++) {
    rank = gridsize_from_cmdline[i] * rank + coords[i];
  }
  return rank;
}

static int lex_rank_from_coords_x(const int *coords, void *fdata)
{
  int rank = coords[3];
  for (int i = 2; i >= 0; i--) {
    rank = gridsize_from_cmdline[i] * rank + coords[i];
  }
  return rank;
}

void QMPInitComms(int argc, char **argv, const int *commDims)
{
  QMP_thread_level_t tl;
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);

  // make sure the QMP logical ordering matches QUDA's
  if (rank_order == 0) {
    int map[] = { 3, 1, 2, 0 };
    QMP_declare_logical_topology_map(commDims, 4, map, 4);
  } else {
    int map[] = { 0, 1, 2, 3 };
    QMP_declare_logical_topology_map(commDims, 4, map, 4);
  }

  CommsMap func = rank_order == 0 ? lex_rank_from_coords_t : lex_rank_from_coords_x;

  initCommsGridQMP(4, commDims, func, NULL);

  int rank = QMP_get_node_number();
	srand(17*rank + 137);

  printf("Rank order is %s major (%s running fastest)\n",  rank_order == 0 ? "column" : "row", rank_order == 0 ? "t" : "x");

}

void QMPFinalizeComms()
{
  QMP_finalize_msg_passing();
}
