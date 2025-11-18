# KV 缓存详解：优化 Transformer 推理效率

> 翻译自 https://huggingface.co/blog/not-lain/kv-caching

[社区文章](/blog/community) 发布于 2025 年 1 月 30 日

## 引言

当 AI 模型生成文本时，它们经常重复许多相同的计算，这会减慢速度。键-值（Key-Value）缓存是一种技术，通过记住先前步骤的重要信息来加速这一过程。模型无需从头重新计算所有内容，而是重用已计算的结果，从而使文本生成更快、更高效。

在本文中，我们将以易懂的方式分解 KV 缓存，解释其为什么有用，并展示它如何帮助 AI 模型更快地工作。

## 先决条件

要完全理解本文内容，读者应熟悉以下内容：

1. Transformer 架构：熟悉注意力机制等组件。
2. 自回归建模：理解像 GPT 这样的模型如何生成序列。
3. 线性代数基础：矩阵乘法和转置等概念，这些对于理解注意力计算至关重要。

这篇[博客](https://huggingface.co/blog/not-lain/tensor-dims) 应该涵盖了本文所需的大部分先决条件。

点击这里查看一些最重要的要点。

* 注意力权重的形状为
* 掩码多头注意力允许每个标记由自身和所有先前标记表示。
* 要生成新标记，模型需要查看所有先前标记及其由先前标记表示的表示

[https://huggingface.co/blog/not-lain/tensor-dims](https://huggingface.co/blog/not-lain/tensor-dims)

## 标准推理与 KV 缓存的兴起

当模型生成文本时，它会查看所有先前标记来预测下一个标记。通常，它会为每个新标记重复相同的计算，这会减慢速度。

KV 缓存通过记住先前步骤的这些计算来解决计算重叠问题，这可以通过在推理过程中存储注意力层的中间状态来实现。

## KV 缓存如何工作？

### 逐步过程

1. 首次生成：当模型看到第一个输入时，它计算并将键和值存储在缓存中。
2. 后续词语：对于每个新词，模型检索存储的键和值，并添加新的键和值，而不是从头开始。
3. 高效注意力计算：使用缓存和新（查询）计算注意力以得出输出。
4. 更新输入：将新生成的标记添加到输入中，直到生成完成。

下面的图示说明了这一过程：

<img src="https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/DbL2RbXFRoMWA5CrOaGB8.png">

```
标记 1: [K1, V1] → 缓存: [K1, V1]
标记 2: [K2, V2] → 缓存: [K1, K2], [V1, V2]
...
标记 n: [Kn, Vn] → 缓存: [K1, K2, ..., Kn], [V1, V2, ..., Vn]
```

KV 缓存
标准推理

在上表中，我们使用了 为了更好的视觉效果，请注意这个数字可以比我们呈现的要大得多。

## 比较：KV 缓存 vs. 标准推理

以下是 KV 缓存与常规生成的比较：

| 特性 | 标准推理 | KV 缓存 |
|---------|--------------------|------------|
| 每个词的计算 | 模型为每个词重复相同的计算。 | 模型重用过去的计算以获得更快的结果。 |
| 内存使用 | 每个步骤使用较少的内存，但随着文本变长，内存会增长。 | 使用额外的内存来存储过去的信息，但保持高效。 |
| 速度 | 随着文本变长而变慢，因为它重复工作。 | 即使文本变长，也保持快速，通过避免重复工作。 |
| 效率 | 计算成本高，响应时间慢。 | 更快、更高效，因为模型记住过去的工作。 |
| 处理长文本 | 由于重复计算，长文本会挣扎。 | 完美适合长文本，因为它记住过去步骤。 |

KV 缓存在速度和效率上带来了巨大差异，特别是对于长文本。通过保存和重用过去计算，它避免了每次从头开始的需要，使其比常规文本生成方式快得多。

## 实际实现

这是一个在 PyTorch 中实现 KV 缓存的简化示例：

```
# PyTorch 中 KV 缓存的伪代码
class KVCache:
    def __init__(self):
        self.cache = {"key": None, "value": None}

    def update(self, key, value):
        if self.cache["key"] is None:
            self.cache["key"] = key
            self.cache["value"] = value
        else:
            self.cache["key"] = torch.cat([self.cache["key"], key], dim=1)
            self.cache["value"] = torch.cat([self.cache["value"], value], dim=1)

    def get_cache(self):
        return self.cache
```

在使用 transformers 库时，这种行为通过 `use_cache` 参数默认启用，您还可以通过 [cache_implementation](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.cache_implementation) 参数访问多种缓存方法，以下是一个简化的代码：

```
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-1.7B')
model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-1.7B').cuda()

tokens = tokenizer.encode("The red cat was", return_tensors="pt").cuda()
output = model.generate(
    tokens, max_new_tokens=300, use_cache = True # 默认设置为 True
)
output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
```

我们在 T4 GPU 上对上述代码进行了基准测试，有/无 KV 缓存，结果如下：

| 使用 KV 缓存 | 标准推理 | 加速 |
|-----------------|--------------------|---------|
| 11.7 秒          | 1 分 1 秒            | ~5.21 倍更快 |

## 结论

KV 缓存是一种简单但强大的技术，有助于 AI 模型更快、更高效地生成文本。通过记住过去计算而不是重复它们，它减少了预测新词所需的时间和努力。虽然它需要额外的内存，但这种方法特别适用于长对话，确保快速高效的生成。

理解 KV 缓存可以帮助开发者和 AI 爱好者构建更快、更智能、更可扩展的语言模型，用于现实世界应用。

我特别感谢 [Aritra Roy Gosthipaty](https://hf.co/ariG23498) 对本文的宝贵支持、反馈和奉献。

## 参考文献与进一步阅读

1. [Transformers KV Caching Explained](https://medium.com/@joaolages/kv-caching-explained-276520203249)
2. [Transformers Key-Value Caching Explained](https://neptune.ai/blog/transformers-key-value-caching)
3. [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
4. [Hugging Face 文档 - Transformers 中的 KV 缓存](https://huggingface.co/docs/transformers/main/en/generation_strategies#kv-caching)

