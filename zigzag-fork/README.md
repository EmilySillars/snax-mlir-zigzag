# SNAX-MLIR-ZigZag Forked Repo
Run and Test ZigZag-Tiled MLIR examples with Verilator Simulating SNAX + gemm.

## Examples

| Test + Description                                           | Matrix Size | Allocation | Values | Tiling Method                                                | Verilator |
| :----------------------------------------------------------- | ----------- | ---------- | ------ | ------------------------------------------------------------ | --------- |
| **Matmul 16x16**<br />Dispatches work to gemm accelerator, which has an 8x8x8 MAC array<br />Full details [here](../kernels/matmul_16x16/matmul_16x16.md) | 16x16       | static     | fixed  | ZigZag w/ [SNAX+gemm](https://github.com/EmilySillars/zigzag/blob/manual-examples/zigzag/inputs/hardware/gemm.yaml) | TODO      |

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

2. ```
   ld.lld-17: error: undefined symbol: memset
   >>> referenced by main.c
   >>>               main.o:(main)
   ```

   Solution: Use `-fno-builtin-memset` or `-fno-builtin`, so edited Makefile.rules's CLAGS macro to include `CFLAGS += -fno-builtin-memset`, and LDFLAGS to include `LDFLAGS += -fno-builtin-memset`!

3. 

4. another error?
