#pragma once

#define QMP_MAX_DIM 6

#ifdef __cplusplus
extern "C" {
#endif

// similar to what is declared in quda.h:
typedef int (*CommsMap)(const int *coords, void *fdata);

typedef struct {
  int ndim;
  int dims[QMP_MAX_DIM];
} LexMapData;

typedef struct Topology_s {
  int ndim;
  int dims[QMP_MAX_DIM];
  int *ranks;
  int (*coords)[QMP_MAX_DIM];
  int my_rank;
  int my_coords[QMP_MAX_DIM];

} Topology;

int comm_rank(void);

int comm_size(void);

int comm_dim(int dim);

void QMPInitComms(int argc, char **argv, const int *commDims);

void QMPFinalizeComms();

#ifdef __cplusplus
}
#endif
