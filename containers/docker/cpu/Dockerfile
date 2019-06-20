## SHOULD BE RUN FROM TOP LEVEL OF REPOSITORY!
## From the base folder of alsvinn, run
#
#    docker build . -f containers/docker/cpu/Dockerfile -t <whatever tag>
#

FROM alsvinn/cpu_base


ENV ALSVINN_IN_DOCKER 1

COPY . /alsvinn


RUN cd /alsvinn &&\
    export PATH=$HOME/local/bin:$PATH:$HOME/local/bin &&\
    mkdir build_docker &&\
    cd build_docker &&\
    cmake ..\
    	  -DCMAKE_BUILD_TYPE=Release \
	  -DALSVINN_PYTHON_VERSION=3.7 \
	  -DCMAKE_PREFIX_PATH=$INSTALL_PREFIX \
	  -DALSVINN_USE_CUDA=OFF && \
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
ENTRYPOINT ["alsuqcli"]