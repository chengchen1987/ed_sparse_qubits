# Intel compiler
ICC_c = icc
ICC_cpp = icpc
ICC_CCFLAGS =  -O3 -qopenmp -mkl 
ICC_LDFLAGS = -fopenmp -L/public1/soft/intel/2019/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64

# Set compile environment
##########################################################################
# Programming language {c/cpp}
PLANG=cpp
# Set compiler {ICC/GNU}
COM=ICC
# The name of executable file
EXECNAME = ed_exHarper
########################################################################

# CPU compiler
CC = ${${COM}_${PLANG}}
# CPU compiling options
CCFLAGS = ${${COM}_CCFLAGS}
# CPU linking options
LDFLAGS = ${${COM}_LDFLAGS}
# The language of sources
CCOBJS := $(patsubst %.${PLANG},%.o,$(wildcard *.${PLANG}))

$(EXECNAME):${CCOBJS}
	${CC} -o $(EXECNAME) ${CCFLAGS} ${CCOBJS} ${LDFLAGS}
include depend
.${PLANG}.o:
	${CC} -c $< $(CCFLAGS) -o $@ 

depend:
	${CC} -MM ${CCOBJS:.o=.${PLANG}} ${CCFLAGS}> depend

clean:
	rm *.o depend
