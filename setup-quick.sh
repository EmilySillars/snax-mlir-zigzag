
# Step 0: get the docker version from the continuous integrationt tests, then launch docker.
# DOCKER_VERSION=$(cat .github/workflows/build-run-kernel.yml | grep -o "snax:v[0-9]\(.[0-9]\)\+") && 
# docker run -itv `pwd`:/repo:z ghcr.io/kuleuven-micas/$DOCKER_VERSION

pip install -e /repo
cd /repo
pip install -r requirements.txt
