#!/bin/bash
scripts = "train_efficientnet_v2m_model_01.py train_efficientnet_v2m_model_02.py train_efficientnet_v2m_model_03.py train_efficientnet_v2m_model_04.py train_efficientnet_v2m_model_05.py"
for script in ${scripts}; do 
    echo "Running ${script}"
    python ${script}
    echo "Complete ${script}"
done
echo "Finished"
