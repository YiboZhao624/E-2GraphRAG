python raptorbaseline.py \
    --dataset NarrativeQA \
    --output_folder /root/projects/lightTAG/NarrativeQA/Raptor \
    --llm_name /root/shared_planing/LLM_model/Qwen2.5-14B-Instruct \
    --embedding_model_name BAAI/bge-m3 \
    --qa_device cuda:0 \
    --summary_device cuda:2 \
    --embedding_device cuda:6 \
