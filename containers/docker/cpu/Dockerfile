## SHOULD BE RUN FROM TOP LEVEL OF REPOSITORY!
## From the base folder of alsvinn, run
#
#    docker build . -f containers/docker/cpu/Dockerfile -t <whatever tag>
#

FROM alsvinn/cpu_base:release-v2.2.0

ARG ALSVINN_USE_FLOAT=OFF
ARG ALSVINN_DOCKER_CONTAINER="Unknown"
ENV ALSVINN_IN_DOCKER 1

COPY . /alsvinn


RUN cd /alsvinn &&\
    export PATH=$HOME/local/bin:$PATH:$HOME/local/bin &&\
    mkdir build_docker &&\
    cd build_docker &&\
    cmake ..\
    	  -DCMAKE_BUILD_TYPE=Release \
	  -DALSVINN_PYTHON_VERSION=${ALSVINN_PYTHON_VERSION} \
	  -DCMAKE_PREFIX_PATH=$INSTALL_PREFIX \
	  -DALSVINN_USE_CUDA=OFF\
	  -DALSVINN_USE_FLOAT=${ALSVINN_USE_FLOAT}\
	  -DALSVINN_DOCKER_CONTAINER=${ALSVINN_DOCKER_CONTAINER}\
	  -DALSVINN_IN_DOCKER=ON && \
    make && \
    make install
    
# Examples easily accesable
RUN cp -r alsvinn/examples /examples &&\
    find /examples -name '*.xml' -exec sed -i 's/platform>cuda/platform>cpu/g' {} \;

# We also want to compile the examples
RUN cd /alsvinn &&\
    cd library_examples/alsuq &&\
    cd only_statistics &&\
    mkdir build_docker && \
    cd build_docker && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} &&\
    make && \
    cd ../.. && \
    cd generate_parameters && \
    mkdir build_docker && \
    cd build_docker && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} &&\
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

ENV PYTHONPATH "${PYTHONPATH}:${INSTALL_PREFIX}/lib/python${ALSVINN_PYTHON_VERSION}/site-packages"
ENTRYPOINT ["alsuqcli"]
