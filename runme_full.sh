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
echo "*          Section 5.3.1  non-batch query          *"
echo "*                                                  *"
echo "****************************************************"
echo "(This may take a while)"
make exec DOCKER_CMD='eval/1-simple-topk.py'
echo "===============  Plotting Figure  7  ==============="
echo "(This may take a while)"
make exec DOCKER_CMD='eval/plot-1-single-query-all.py'
echo "===============  Plotting Figure  8  ==============="
make exec DOCKER_CMD='eval/plot-1-single-query-part.py'


echo -e "\n"
echo "****************************************************"
echo "*                                                  *"
echo "*          Section 5.3.2  batch query (a)          *"
echo "*                                                  *"
echo "****************************************************"
make exec DOCKER_CMD='eval/2-batched-a.py'
echo "=============  Plotting Figure  9 (a)  ============="
make exec DOCKER_CMD='eval/plot-2-batched-a.py'

echo -e "\n"
echo "****************************************************"
echo "*                                                  *"
echo "*          Section 5.3.2  batch query (b)          *"
echo "*                                                  *"
echo "****************************************************"
make exec DOCKER_CMD='eval/2-batched-b.py'
echo "=============  Plotting Figure  9 (b)  ============="
make exec DOCKER_CMD='eval/plot-2-batched-b.py'

echo -e "\n"
echo "****************************************************"
echo "*                                                  *"
echo "*          Section 5.3.2  batch query (c)          *"
echo "*                                                  *"
echo "****************************************************"
make exec DOCKER_CMD='eval/2-batched-c.py'
echo "=============  Plotting Figure  9 (c)  ============="
make exec DOCKER_CMD='eval/plot-2-batched-c.py'

echo -e "\n"
echo "****************************************************"
echo "*                                                  *"
echo "*          Section 5.3.3  ablation study           *"
echo "*                                                  *"
echo "****************************************************"
echo "(This may take a while)"
make exec DOCKER_CMD='eval/4-ablation.py'
echo "===============  Plotting Figure 10  ==============="
make exec DOCKER_CMD='eval/plot-4-ablation.py'

echo -e "\n"
echo "****************************************************"
echo "*                                                  *"
echo "*         Section 5.4  quantile results            *"
echo "*                                                  *"
echo "****************************************************"
make exec DOCKER_CMD='eval/ex1-median.py'
echo "===============  Plotting Figure 11  ==============="
make exec DOCKER_CMD='eval/plot-ex1-median-fig.py'

echo -e "\n"
echo "****************************************************"
echo "*                                                  *"
echo "*         Section 5.5  robustness results          *"
echo "*                                                  *"
echo "****************************************************"
make exec DOCKER_CMD='eval/3-skewed.py'
echo "=========  Plotting Figure 12 (a) and (b)  ========="
make exec DOCKER_CMD='eval/plot-3-skewed.py'
echo "----------------------------------------------------"
make exec DOCKER_CMD='eval/ex2-skewed.py'
echo "=========  Plotting Figure 12 (c) and (d)  ========="
make exec DOCKER_CMD='eval/plot-ex2-skewed-fig.py'
echo "----------------------------------------------------"
echo "(This may take a while)"
make exec DOCKER_CMD='eval/ex3-zipf-a.py'
echo "=============  Plotting Figure 13 (a)  ============="
make exec DOCKER_CMD='eval/plot-ex3-zipf-fig-a.py'
echo "----------------------------------------------------"
echo "(This may take quite a while)"
make exec DOCKER_CMD='eval/ex3-zipf-b.py'
echo "=============  Plotting Figure 13 (b)  ============="
make exec DOCKER_CMD='eval/plot-ex3-zipf-fig-b.py'

echo -e "\n"
echo "****************************************************"
echo "*                                                  *"
echo "*                End of evaluation!                *"
echo "*                                                  *"
echo "****************************************************"
