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

az login --tenant berkeleydatasciw255.onmicrosoft.com
kubectl config use-context w255-aks
IMAGE_PREFIX=$(az account list --all | jq '.[].user.name' | grep -i berkeley.edu | awk -F@ '{print $1}' | tr -d '"' | uniq)

kubectl -n edbrown apply -k ${K8S_PROD_DIR}
ACR_NAME=w255mids
az acr login --name ${ACR_NAME}
docker push ${ACR_NAME}.azurecr.io/lab4:latest