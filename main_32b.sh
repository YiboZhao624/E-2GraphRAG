export CUDA_VISIBLE_DEVICES=6

# 依次运行所有 32b 相关的 LLMExtractor / LLMVerifier 配置
for cfg in manual_fin_configs/{llmextractor,hanlp}_request_llm_config_32b*.yaml; do
  echo "=== Running $cfg ==="
  python -u main.py --config "$cfg"
done

# for cfg in manual_mix_configs/{llmextractor,hanlp}_request_llm_config_32b*.yaml; do
#   echo "=== Running $cfg ==="
#   python -u main_mix.py --config "$cfg"
# done