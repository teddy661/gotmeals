RAW_SCRIPT_DIR="$(dirname "$0")"
SCRIPT_DIR=$(realpath ${RAW_SCRIPT_DIR})
cd ${SCRIPT_DIR}
TOP_LAB_DIR=$(pwd | awk -F "/" '{print $NF}')
DO_DEPLOYMENT=0
PROG_NAME=$(basename ${0})

## Tag the container with the git tag
LAB=./src
GIT_TAG=$(git rev-parse --short --verify HEAD)
if [[ -d ${LAB} ]]; then 
	cat >| ${LAB}/src/app_version.py <<PYTHONCONTENT
def get_app_version():
	return {"git-version": "${GIT_TAG}"}
PYTHONCONTENT
else
	echo "Required directory: ${LAB} doesn't exist. Exiting"
	exit 20
fi

K8S_PROD_DIR=${SCRIPT_DIR}/.k8s/overlays/prod
DEPLOYMENT_TEMPLATE=${K8S_PROD_DIR}/patch-deployment-lab4_copy.yaml
DEPLOYMENT_FILE=${K8S_PROD_DIR}/patch-deployment-lab4.yaml

ACR_NAME=w255mids