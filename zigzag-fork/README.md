# SNAX-MLIR-ZigZag Forked Repo
Run and Test ZigZag-Tiled MLIR examples with Verilator Simulating SNAX + gemm.
### *quick setup*


0. ```
   DOCKER_VERSION=$(cat .github/workflows/build-run-kernel.yml | grep -o "snax:v[0-9]\(.[0-9]\)\+") && 
   docker run -itv `pwd`:/repo:z ghcr.io/kuleuven-micas/$DOCKER_VERSION
   ```

1. ```
   cd repo
   ```

2. ```
   bash setup-quick.sh
   ```
### *quick run*

```
bash run-quick.sh simple_mult
```
## setup

0. ```
   sudo apt install docker.io
   ```

1. ```
   docker run -itv `pwd`:/repo:z ghcr.io/kuleuven-micas/snax:<docker version>
   ```

where `<docker version>` is `v0.1.6` from `image: ghcr.io/kuleuven-micas/snax:v0.1.6` from `.github/workflows/build-run-kernel.yml`

2. ```
   pip install -e /repo
   ```

3. ```
   cd /repo
   pip install -r requirements.txt
   ```

## run example

0. All the set up instructions

1. ```
   cd /repo/kernels/simple_mult
   make allrun
   ```

## trouble shooting

1. ```
   docker: permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/containers/create": dial unix /var/run/docker.sock: connect: permission denied.
   See 'docker run --help'.
   ```

   Solution: [here](https://www.digitalocean.com/community/questions/how-to-fix-docker-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket)

   ```
   sudo chmod 666 /var/run/docker.sock
   ```

2. error 2

