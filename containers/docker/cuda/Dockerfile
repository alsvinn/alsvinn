## SHOULD BE RUN FROM TOP LEVEL OF REPOSITORY!
## From the base folder of alsvinn, run
#
#    docker build . -f containers/docker/cuda/Dockerfile -t <whatever tag>
#

FROM alsvinn/cuda_base:release-v2.2.0

ARG ALSVINN_USE_FLOAT=OFF
ARG ALSVINN_DOCKER_CONTAINER="Unknown"
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
    	  -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX  \
    	  -DCMAKE_C_COMPILER=`which $CC` \
    	  -DCMAKE_CXX_COMPILER=`which $CXX` \
	  -DALSVINN_PYTHON_VERSION=${ALSVINN_PYTHON_VERSION}\
	  -DALSVINN_USE_FLOAT=${ALSVINN_USE_FLOAT} \
	  -DALSVINN_DOCKER_CONTAINER=${ALSVINN_DOCKER_CONTAINER}\
	  -DALSVINN_IN_DOCKER=ON && \
	  
    make && \
    make install
    
# Examples easily accesable
RUN cp -r alsvinn/examples /examples

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
    cp generate_parameters /usr/local/bin/ &&\
    cd ../.. && \
    cd structure_standalone && \
    mkdir build_docker && \
    cd build_docker && \
    cmake .. -DCMAKE_BUILD_TYPE=Release &&\
    make &&\
    cp structure_standalone /usr/local/bin/
    

RUN rm /etc/ld.so.cache && ldconfig
RUN ldconfig
ENTRYPOINT ["alsuqcli"]
ENV PYTHONPATH "${PYTHONPATH}:${INSTALL_PREFIX}/lib/python${ALSVINN_PYTHON_VERSION}/site-packages"