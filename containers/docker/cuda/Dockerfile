## SHOULD BE RUN FROM TOP LEVEL OF REPOSITORY!
## From the base folder of alsvinn, run
#
#    docker build . -f containers/docker/cuda/Dockerfile -t <whatever tag>
#

FROM alsvinn/cuda_base


ENV ALSVINN_IN_DOCKER 1

COPY . /alsvinn


RUN cd /alsvinn &&\
    export PATH=$HOME/local/bin:$PATH:$HOME/local/bin &&\
    mkdir build_docker &&\
    cd build_docker &&\
    $INSTALL_PREFIX/bin/cmake ..\
          -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} \
    	  -DCMAKE_BUILD_TYPE=Release \
    	  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    	  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda	  \
    	  -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython2.7.so.1 \
    	  -DPYTHON_INCLUDE_DIR=/usr/include/python2.7 \
    	  -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX  \
    	  -DCMAKE_C_COMPILER=`which $CC` \
    	  -DCMAKE_CXX_COMPILER=`which $CXX` \
	  -DALSVINN_PYTHON_VERSION=2.7 && \
    make && \
    make install

# We also want to compile the examples
RUN cd /alsvinn &&\
    cd library_examples/alsuq &&\
    cd only_statistics &&\
    mkdir build_docker && \
    cd build_docker && \
    cmake .. -DCMAKE_BUILD_TYPE=Release &&\
    make && \
    cd ../.. && \
    cd generate_parameters && \
    mkdir build_docker && \
    cd build_docker && \
    cmake .. -DCMAKE_BUILD_TYPE=Release &&\
    make &&\
    cp generate_parameters /usr/local/bin/
    

RUN rm /etc/ld.so.cache && ldconfig
RUN ldconfig
ENTRYPOINT ["alsuqcli"]