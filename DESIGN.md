# mrag 设计说明

本文说明系统的整体架构、各模块职责、关键设计决策与已知局限。

---

## 1. 架构总览

```
                         ┌───────────────────────────┐
                         │         main.cc           │
                         │  解析 --ingest / --chat    │
                         └─────────────┬─────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │          MragApp          │
                         │  BackendGuard（最先构造、   │
                         │   最后析构，包住所有引擎）   │
                         └──────┬─────────────┬──────┘
                  buildKnowledgeBase()    chatLoop()
                                │             │
        ┌───────────────────────┘             └───────────────────────┐
        ▼                                                             ▼
┌──────────────────┐                                        ┌──────────────────┐
│ DocumentProcessor│                                        │  EmbeddingEngine │◄─┬─┐
│  读 TXT / 切块    │                                        │   问题 → 向量     │  │ │
└────────┬─────────┘                                        └────────┬─────────┘  │ │
         ▼ vector<Chunk>                                             ▼ query_emb  │ │
┌──────────────────┐   chunk.embedding   ┌──────────────────┐        │         │ │
│ EmbeddingEngine  │◄────────────────────┤   VectorDatabase ├────────┘         │ │
│   文本 → 向量     │                     │  插入 / 余弦检索   │                  │ │
└──────────────────┘                     │  saveToDisk /     │                  │ │
                                         │  loadFromDisk     │                  │ │
                                         └────────┬─────────┘                   │ │
                                                  ▼ top-k Chunk                 │ │
                                         ┌──────────────────┐                   │ │
                                         │ GenerationEngine │───────────────────┘ │
                                         │   流式生成回答     │                     │
                                         └──────────────────┘                     │
                                    两者都继承自 → LlamaModelBase ◄────────────────┘
```

- 实线：数据流（建库走左半，问答走右半）
- 两个引擎都继承 `LlamaModelBase`，复用其模型/上下文的加载与释放

---

## 2. 模块职责

| 文件 | 类 / 职责 |
|---|---|
| `main.cc` | 程序入口。屏蔽 llama.cpp 底层日志（`llama_log_set`）、解析命令行、选择运行模式、异常兜底 |
| `mragapp.h/.cc` | `MragApp`：协调两大流程；`BackendGuard` 内部结构体保证 `llama_backend_init/free` 的配对 |
| `config.h/.cc` | `AppConfig`：从 `config.json` 读参数，字段缺失保留默认值，文件不存在不报错 |
| `document.h/.cc` | `DocumentProcessor`：读 TXT、去 `\r`、识别章节标题、按字节切块（对齐 UTF-8 边界、优先落在句末标点）、生成 overlap |
| `chunk.h/.cc` | `Chunk`：数据载体（id / text / metadata / embedding）+ 二进制序列化/反序列化 |
| `vectordatabase.h/.cc` | `VectorDatabase`：插入、余弦相似度 top-k 检索（最小堆）、带魔数/版本号的持久化 |
| `llamamodel.h/.cc` | `LlamaModelBase`：RAII 管理 `llama_model*` / `llama_context*`，禁用拷贝；提供 `tokenize()` |
| `embeddingengine.h/.cc` | `EmbeddingEngine`：`sanitizeUtf8` 清理输入 + 单次前向取出句向量 |
| `generationengine.h/.cc` | `GenerationEngine`：组装 ChatML prompt + prefill + 采样链 + 自回归流式生成 |

**依赖方向**：上层的类依赖下层的数据结构，但不反向依赖。`Chunk` 是最底层的纯数据，
被 `document` / `vectordatabase` 共享。

---

## 3. 关键设计决策

### 3.1 RAII 管理 llama.cpp 资源

`llama_model*` 与 `llama_context*` 是 C 风格的不透明句柄，必须手动释放。`LlamaModelBase`
把它们包在 C++ 对象里：

- 构造函数：加载模型、创建上下文，失败抛 `std::runtime_error`（带路径）
- 析构函数：**先释放 `ctx_`，再释放 `model_`**（上下文依赖模型，顺序不可颠倒）
- 构造函数中若上下文创建失败，需**手动释放已加载的模型**再抛异常
  （此时析构函数不会被调用，否则泄漏）

### 3.2 Rule of Five —— 禁用拷贝

两个对象若共享同一份 `llama_model*`，析构时会 **double free**。因此：

```cpp
LlamaModelBase(const LlamaModelBase&)            = delete;
LlamaModelBase& operator=(const LlamaModelBase&) = delete;
```

这也意味着两个引擎**只能在初始化列表中构造**，不能赋值。

### 3.3 `BackendGuard` —— 用构造/析构顺序做事

`llama_backend_init/free` 必须成对，且包住所有模型操作。做法是把它们放进一个内部结构体，
并**在成员列表里声明在最前**：

```cpp
struct BackendGuard { BackendGuard(); ~BackendGuard(); };  // init / free
BackendGuard guard_;     // 声明最前 → 最先构造、最后析构
AppConfig    cfg_;
EmbeddingEngine emb_;    // 在 guard_ 之后构造 → 被它包住
GenerationEngine gen_;
```

利用 C++「成员按声明顺序构造、按反序析构」的保证，无需任何显式调用。

### 3.4 二进制序列化格式

**单个 `Chunk`**（`Chunk::serialize` / `deserialize`）：

```
[id: uint64] [text_len: uint64] [text ...]
             [meta_len: uint64] [metadata ...]
             [emb_dim:  uint64] [emb_dim 个 float]
```

**数据库文件**（`VectorDatabase::saveToDisk` / `loadFromDisk`）：

```
[magic: 0x4D524147] [version: 1] [count: uint64] [chunk0] [chunk1] ...
```

- 用一个「魔数 + 版本号」的头，加载时**先校验**，不匹配就拒绝并报错。
- 变长字段（字符串、向量）一律**先写长度再写内容** —— 否则读的时候无法知道边界。
- 长度统一用 `uint64_t`，避免 `size_t` 在不同平台宽度不一致。

### 3.5 切块：UTF-8 边界与标点对齐

切块按**字节**阈值进行，但有两个约束：

1. **不截断多字节字符** —— `nextUtf8Boundary` 把任意字节位置修正到最近的字符起点。
   UTF-8 的续字节一律形如 `10xxxxxx`，据此可判断「某位置是不是字符起点」。
   > 注意：不能依赖 `std::regex` 的字符类来做多字节判断 —— `std::regex<char>` 是**按字节**
   > 工作的，`[章节回卷]` 会被拆成字节集合，误匹配大量汉字。章节识别改为用
   > `std::string::find` / `compare` 做**字节精确**匹配。
2. **优先落在句末标点** —— 用 `rfind` 在切点附近找 `。！？` 或换行，退不到
   `overlap_size` 以内时放弃对齐，直接按字节切（防止 `pos - overlap` 无符号下溢）。

### 3.6 检索：最小堆求 top-k

`search` 用**容量为 k 的小顶堆**：遍历所有 chunk，算余弦相似度后压入，堆超过 k 就弹出堆顶
（当前最小的）。这样堆里始终保留相似度最大的 k 个，复杂度 **O(n log k)**。

- 小顶堆的堆顶是最小值 —— 正好用来「踢掉最不相关的」。
- 结果按相似度降序返回。

### 3.7 Prompt：ChatML 格式

生成引擎把检索到的片段按模板拼成单轮对话：

```
<|im_start|>system
你是一个严谨的小说阅读助手。……（只能依据参考上下文回答）
<|im_start|>user
[参考上下文]
【片段 0 | 第一章 秋天的信】
……
用户问题：xxx<|im_end|>
<|im_start|>assistant
```

- 每个片段带**出处标注**（编号 + 章节名），片段之间空行分隔。
- 分词时 `parse_special = true`，让 `<|im_start|>` 等标记成为**真正的控制 token**，
  否则它们会被当成普通文本，模型也会把结束符当文字吐出来。

### 3.8 生成：采样链与自回归

- 采样链顺序：`top_k → top_p → temperature → dist`（顺序影响结果，不可颠倒）。
- 每步：`llama_sampler_sample(smpl, ctx, -1)` 采样 → 判 EOG → `llama_token_to_piece` 转文字
  → `std::cout << … << std::flush`（流式）→ 把新 token 组成单 token batch → `llama_decode`。
- 注意：`llama_batch_init` 分配的 batch **需要** `llama_batch_free`；而
  `llama_batch_get_one` 包装的是外部指针，**不需要**释放。

---

## 4. 数据流

**建库（`--ingest`）**

```
TXT 文件 → DocumentProcessor.processNovel
        → vector<Chunk>（每块带章节 metadata）
        → 逐块 EmbeddingEngine.generateEmbedding → 写入 chunk.embedding
        → VectorDatabase.insert
        → VectorDatabase.saveToDisk
```

**问答（`--chat`）**

```
VectorDatabase.loadFromDisk
   ↓ 循环读入问题
EmbeddingEngine.generateEmbedding(问题) → query 向量
   ↓
VectorDatabase.search(query, top_k) → 最相关的 k 个 Chunk
   ↓
GenerationEngine.generateStream(问题, chunks) → 流式回答
```

---

## 5. 已知局限

1. **生成质量受限于小模型** —— 1.5B 参数的 qwen 对复杂问题容易答得简略或不准。
2. **切块按字节而非语义** —— 可能把一句话切到两块；已通过标点对齐缓解，但非完美。
3. **章节识别依赖固定格式** —— 仅匹配「第…章/节/回/卷」或「Chapter N」开头的行。
4. **检索为全量暴力计算** —— 规模上万后变慢（未引入近似最近邻索引）。
5. **无多轮对话** —— 每轮问答独立，不保留对话历史。
6. **单文档建库** —— 一次 `--ingest` 只处理一个 TXT（多文档需要多次建库或自行合并）。
7. **`sanitizeUtf8` 为简单实现** —— 逐字节校验序列完整性，丢弃非法序列，不做替换字符。
