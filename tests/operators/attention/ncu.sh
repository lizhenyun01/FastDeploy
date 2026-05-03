

ncu --clock-control=reset

ncu  --target-processes all --set full -f -o attn_v18_all -k regex:"multi_query_append_attention_warp1_4_kernel|decode_append_attention_c16_kernel" python tests/operators/attention/benchmark_decode_attention.py --op both
