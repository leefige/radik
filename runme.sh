#! /bin/bash

# build image
make build

# check GPU status
echo -e "\n"
echo "****************************************************"
echo "*                                                  *"
echo "*                 GPU status check                 *"
echo "*                                                  *"
echo "****************************************************"
make exec DOCKER_CMD='nvidia-smi'

# run evaluation scripts
echo -e "\n"
echo "****************************************************"
echo "*                                                  *"
echo "*          Section 5.3.2  batch query (a)          *"
echo "*                                                  *"
echo "****************************************************"
make exec DOCKER_CMD='eval/2-batched-a.py'
echo "=============  Plotting Figure 9 (a)  ============="
make exec DOCKER_CMD='eval/plot-2-batched-a.py'

echo -e "\n"
echo "****************************************************"
echo "*                                                  *"
echo "* For full evaluation, please run 'runme_full.sh'  *"
echo "*                                                  *"
echo "****************************************************"
