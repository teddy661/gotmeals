#!/bin/bash
###############################################################################
###############################################################################
##
##
## Build Run Predict Container in Minikube
##
##
###############################################################################
###############################################################################
RAW_SCRIPT_DIR="$(dirname "$0")"
SCRIPT_DIR=$(realpath ${RAW_SCRIPT_DIR})
cd ${SCRIPT_DIR}
TOP_LAB_DIR=$(pwd | awk -F "/" '{print $NF}')
HOSTNAME="127.0.0.1"
PROG_NAME=$(basename ${0})

display_usage()
{
	echo "Usage: ${PROG_NAME} [-h|-d] "
	echo -e "\t{no option runs minikube, leaves it running and starts the tunnel}"
	echo ""
	echo "-h"
	echo -e "\t print this help message"
	echo "-d"
	echo -e "\t clean and stop a running environment if left running with -r flag"
}

while getopts "htmrd" option; do
	case ${option} in
		h)
			display_usage
			exit 0
			;;
		d)
			CLEAN_AND_STOP_ONLY=1
			;;
		*)
			display_usage
			exit 6
			;;
	esac
done

clean_and_stop()
{
	echo -e "================================================================================"
	echo -e "${PROG_NAME}: INFO: Clean up after yourself\n"
	GOT_MATCH=0
	if [[ ! -z ${MK_TUNNEL_PID} ]]; then
		GOT_MATCH=1
	else
		# Should only be one tunnel once we get a match quit and move on 
		PS_PID=$(ps -ef | egrep minikube | egrep -v egrep | awk '{print $2}')
		if [[ ! -z ${PS_PID} ]]; then 
			for PID in ${PS_PID}
			do
				egrep tunnel /proc/${PID}/cmdline  > /dev/null
				if [[ ${?} -eq 0 ]]; then 
					MK_TUNNEL_PID=${PID}
					GOT_MATCH=1
					break
				fi
			done
		fi		
	fi
	if [[ ${GOT_MATCH} -eq 0 ]]; then
		echo -e "tunnel not running"
	else
		echo -e "stopping tunnel"
		kill ${MK_TUNNEL_PID}
	fi
	MK_HOST_STATUS=$(minikube status  | grep host | awk '{print $2}')
	if [[ ${MK_HOST_STATUS} != 'Stopped' ]]; then 
        minikube kubectl -- config use-context minikube
		minikube kubectl -- -n edbrown delete deployments,svc --all --wait=true
		minikube kubectl -- -n edbrown delete statefulsets  --all --wait=true
		minikube kubectl -- -n edbrown delete horizontalpodautoscaler.autoscaling/lab4 --wait=true
		minikube stop
	else
		echo -e "minikube was not running"
	fi
}


if [[ ! -z ${CLEAN_AND_STOP_ONLY} ]]; then
	clean_and_stop
	exit 0
fi

## Tag the container with the git tag
GIT_TAG=$(git rev-parse --verify HEAD)
if [[ -d ./src ]]; then 
	cat >| ./src/app_version.py <<PYTHONCONTENT
def get_app_version():
	return {"git-version": "${GIT_TAG}"}
PYTHONCONTENT
else
	echo "Required directory: ./src doesn't exist. Continuing"
fi

docker context use default 

minikube start --kubernetes-version=v1.27.3 --memory=32768 --cpus=8

eval $(minikube docker-env)

K8S_DEV_DIR="${SCRIPT_DIR}/.k8s/overlays/dev"

minikube kubectl -- -n edbrown apply -k ${K8S_DEV_DIR}
minikube kubectl -- -n edbrown create secret generic elasticsearch-pw --from-literal=ELASTIC_PASSWORD="jFoEj4A&5dnrCrQm"

echo -e "================================================================================"
echo -e "${PROG_NAME}: INFO: Start Minikube Tunnel\n"
minikube tunnel &
MK_TUNNEL_PID=$!

echo "Waiting for Elasticsearch Tunnel to Connect"
until [ \
	"$(minikube kubectl -- -n edbrown get svc | grep LoadBalancer | grep elasticsearch | awk '{print $4}')" \
	== "127.0.0.1" ]
	do
	sleep 1
	((i++))
	if [[ ${i} == 10 ]]; then 
		echo "Tunnel no detected after ${i} seconds. Giving up"
		exit 5
	fi
	done
echo "Elasticsearch Tunnel Connected"
echo -e ""

echo "Waiting for predict Tunnel to Connect"
until [ \
	"$(minikube kubectl -- -n edbrown get svc | grep LoadBalancer | grep lab4 | awk '{print $4}')" \
	== "127.0.0.1" ]
	do
	sleep 1
	((i++))
	if [[ ${i} == 10 ]]; then 
		echo "Tunnel no detected after ${i} seconds. Giving up"
		exit 5
	fi
	done
echo "Predict Tunnel Connected"
echo -e ""


echo -e "================================================================================"
echo -e "${PROG_NAME}: INFO: Wait for your API to become accessible\n"
minikube kubectl -- -n edbrown  rollout status -w deployment/elasticsearch
minikube kubectl -- -n edbrown  rollout status -w deployment/lab4
echo -e ""