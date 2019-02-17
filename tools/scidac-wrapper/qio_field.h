#pragma once

//header file with declarations from QUDA library

#ifdef __cplusplus
extern "C" {
#endif

void read_gauge_field(const char *filename, void *gauge, const int prec, const int *X,
		      int argc, char *argv[]);
void write_gauge_field(const char *filename, void* gauge, const int prec, const int *X,
          int argc, char* argv[]);
void read_spinor_field(const char *filename, void *V[], const int precision, const int *X,
		       int nColor, int nSpin, int Nvec, int argc, char *argv[]);
void write_spinor_field(const char *filename, void *V[], const int precision, const int *X,
			int nColor, int nSpin, int Nvec, int argc, char *argv[]);

#ifdef __cplusplus
}
#endif
