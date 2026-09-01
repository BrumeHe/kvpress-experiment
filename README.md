# 实验：kvpress 中不同算法随kv预算变化的困惑度测试

模型：Qwen2.5-1.5B-Instruct（28 层，GQA 12q/2kv，bf16，eager attention）


长文本：LongBench narrativeqa 前 9216 token = 前缀 6144 + 评测段 3072


计算资源: RTX 4060 laptop 8GB


主体部分代码: scripts/


结果: results/


前缀 prefill 时用 press 压缩到预算 B；评测段逐 chunk 前向，cache = 压缩前缀 + 已有评测 token（评测 KV 保留，与 chunk 大小无关）



共测试了 full random 和kvpress中的：streamingllm snapkv PyramidKV ObservedAttention（kvpress中关于H2O论文的类似实现）


random的结果与kvpress算法差距较大（困惑度在KV预算低时很高，仅在KV预算4096时较低），破坏可视化比例未呈现在图中，可以参考ppl_results.csv数据


![alt text](ppl_plot.png)