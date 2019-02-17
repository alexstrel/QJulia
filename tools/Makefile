CXX = mpicxx
CC  = mpicc

COPT  = -fPIC -std=c99 -O3 -Wall -fopenmp
CXXOPT  = -fPIC -std=c++11 -O3 -Wall -fopenmp

EIGEN_DIR=/opt/eigen-3.3.7/include/eigen3
SCIDAC_DIR=/opt/scidac-gcc-7.3.0-ompi-3.1.2

EIGENWRAPPER_DIR=./eigen-wrapper
SCIDACWRAPPER_DIR=./scidac-wrapper

EIGENWRAP_INC = -I$(EIGENWRAPPER_DIR) -I$(EIGEN_DIR)
SCIDACWRAP_INC = -I$(SCIDACWRAPPER_DIR) -I$(SCIDAC_DIR)/include

SCIDACLDFLAG = -shared -Wl,-soname,libscidacwrap.so.1 -L${SCIDAC_DIR}/lib -lqio -lqmp -llime
EIGENLDFLAG = -shared -Wl,-soname,libeigpc.so.1

all: libeigenpc.so.1.0 libscidacwrap.so.1.0

libeigenpc.so.1.0: $(EIGENWRAPPER_DIR)/eigenpc.o
	$(CXX) -o  $@  $^  $(EIGENLDFLAG)   

$(EIGENWRAPPER_DIR)/eigenpc.o : $(EIGENWRAPPER_DIR)/eigenpc.cpp
	$(CXX) $(CXXOPT) -c -o  $@  $?  $(EIGENWRAP_INC)  

libscidacwrap.so.1.0: $(SCIDACWRAPPER_DIR)/qio_field.o $(SCIDACWRAPPER_DIR)/qio_util.o $(SCIDACWRAPPER_DIR)/qmp_util.o $(SCIDACWRAPPER_DIR)/layout_hyper.o
	$(CXX) -o  $@  $^  $(SCIDACLDFLAG)   

$(SCIDACWRAPPER_DIR)/%.o : $(SCIDACWRAPPER_DIR)/%.cpp
	$(CXX) $(CXXOPT) -c -o  $@  $?  $(SCIDACWRAP_INC) 

$(SCIDACWRAPPER_DIR)/%.o : $(SCIDACWRAPPER_DIR)/%.c
	$(CC) $(COPT) -c -o  $@  $?  $(SCIDACWRAP_INC)

copy:
	mv libeigenpc.so.1.0 $(EIGENWRAPPER_DIR)
	ln -sf libeigenpc.so.1.0 $(EIGENWRAPPER_DIR)/libeigenpc.so
	mv libscidacwrap.so.1.0 $(SCIDACWRAPPER_DIR)
	ln -sf libscidacwrap.so.1.0 $(SCIDACWRAPPER_DIR)/libscidac.so

clean:
	rm $(EIGENWRAPPER_DIR)/*.o $(EIGENWRAPPER_DIR)/*.so.* 
	rm $(SCIDACWRAPPER_DIR)/*.o $(SCIDACWRAPPER_DIR)/*.so.*

.PHONY:	clean	

