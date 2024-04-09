#!/bin/bash
scripts="baseline_efficientnet_b7.py baseline_efficientnet_v2l.py baseline_efficientnet_v2m.py baseline_vgg19.py fine_tune_efficientnet_b7.py fine_tune_efficientnet_v2l.py fine_tune_efficientnet_v2m.py fine_tune_vgg19.py" 
for script in ${scripts}; do
    echo "Running ${script}"
    python ${script}
    echo "Complete ${script}"
done
echo "Finished"
