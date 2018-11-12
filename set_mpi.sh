#!/usr/bin/bash

export JULIA_MPI_C_COMPILER=/opt/openmpi-3.1.2-gcc-7.3.0-dyn/bin/mpicc
export JULIA_MPI_Fortran_COMPILER=/opt/openmpi-3.1.2-gcc-7.3.0-dyn/bin/mpif90
export JULIA_MPI_C_LIBRARIES="-L/opt/openmpi-3.1.2-gcc-7.3.0-dyn/lib -lmpi"
export JULIA_MPI_Fortran_INCLUDE_PATH="-I/opt/openmpi-3.1.2-gcc-7.3.0-dyn/include"
export JULIA_MPI_Fortran_LIBRARIES="-L/opt/openmpi-3.1.2-gcc-7.3.0-dyn/lib -lmpi_usempif08 -lmpi_mpifh -lmpi"

