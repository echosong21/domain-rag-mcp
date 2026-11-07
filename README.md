# domain-rag-mcp

把领域可配置的 RAG 检索能力包装成标准 MCP 工具，**并按 scope 做渐进披露**。

一行启动：

```bash
python server.py --profile health-agent
```

---

## 一、这个系统解决什么问题

MCP（Model Context Protocol）让 Agent 能调用外部工具。但直接接上会遇到三个问题：

**① 工具一多，Agent 就会被淹没。** 一个 RAG 服务可能有十几个能力（检索、索引状态、重建、
安全闸、评测……），全塞进 `tools/list` 会让模型在无关工具上浪费注意力。
更糟的是**它可能尝试无权使用的工具**，得到一连串拒绝，然后学会"乱试"。

**② 不同 Agent 该看到不同工具。** 面向用户的产品 Agent 不该看到「重建索引」，
更不该看到「开发期评测」。这不是"调不动就行"——**看不见**比**调不动**更安全。

**③ MCP 规范不允许按连接维护状态。** 2026-07-28 版移除了协议级 session，
且规定 `tools/list` 不得随连接变化 —— 但**允许它因授权不同而不同**。

**这个项目就是围绕第三点设计的**：把「已披露工具」等价于「已授予 scope 的工具」，
状态按 principal（Agent 身份）存，而不是按连接存。

---

## 二、核心设计：`enable` 是授予 scope，不是改会话状态

这是整个项目最反直觉、也最关键的一点。

**看起来**：`meta.enable_tool("common.ops.reindex")` 让一个工具变得可见。
**实际上**：它把 `common.ops.write` 这个 **scope 授予 principal**，然后该 scope 下的
**所有**工具一起变为可见。

```
enable_tool(tool_name) → 找到该 tool 的 scope → 若在 discovery_scopes 内
                       → 加入 principal 的已授予 scope 集
                       → 该 scope 下所有工具 enable()
                       → 自动发 tools/list_changed
```

**为什么不是按工具授予**：权限的原子单位是 scope，不是 tool。
一个 scope 覆盖多个工具（如 `common.safety` 覆盖 `get_rules` / `check_input` /
`check_output`），按工具授予会出现「同一个权限域内部分可见」的畸形状态 ——
既不好解释，也不好审计。

**为什么这个状态能跨连接存在**：因为它**不属于连接**。连接断了，
principal 的授权还在；重连后 `tools/list` 返回同一个结果。这正是规范允许的
「因授权不同而不同」。

---

## 三、多 Agent 隔离：两个 profile 的实测差异

同一个 server，换 `--profile` 就得到不同的可见工具集。**这是权限矩阵真的在起作用的
直接证据**，不是设计意图。

| | `health-agent` | `dev-agent` |
|---|---|---|
| **初始 scope** | 4 个（无写权限） | 5 个（含 `common.ops.write`） |
| **可发现 scope** | 4 个 —— **不含**写权限与 `dev.*` | **7 个全含** |
| **可用领域** | `health` | `health` + `law` |
| **写操作开关** | `allow_ops_write: false` | `true` |

**关键差异在第二行**：`health-agent` 的 `discovery_scopes` 刻意不含
`common.ops.write` 与 `dev.*`。这意味着**它不可能通过 `enable_tool` 提权** ——
搜索搜不到，自然也就申请不到。

**这是双重保险**：

1. **scope 过滤** —— 搜不到就申请不到
2. **`allow_ops_write` 开关** —— 即使通过某种方式拿到了 scope，`reindex` 仍会被拒

**⇒ 权限隔离不靠"记得检查"，靠两层独立的机制。**

想看差异：

```bash
python server.py --profile health-agent --list-tools
python server.py --profile dev-agent   --list-tools
```

---

## 四、架构

```
┌─ profiles/*.yaml —— 每个 Agent 一个授权声明 ───────────────┐
│  接入新 Agent = 写一个 yaml，核心代码不动                   │
├─ registry/registry.yaml —— 工具注册表（唯一事实源）─────────┤
│  加一个工具 = 这里加一段 + 新建一个 py 文件，核心代码不动     │
├─ disclosure/ —— 披露层 ───────────────────────────────────┤
│  principal（授权状态）· 元工具（搜索/申请）· 可见性计算       │
├─ tools/ —— 工具实现（按 scope 分目录）──────────────────────┤
│  common.discovery / common.knowledge / common.safety / ...  │
└─ observability/ · resilience/ · transport/ ────────────────┘
   审计与指标 · 降级与熔断 · 传输层
```

**三层分离的意义**：

- **profiles 是薄的** —— 只是授权声明，没有逻辑
- **registry 是事实源** —— 工具清单、scope 归属、降级链都在这里
- **代码是厚的** —— 披露逻辑、权限校验、审计都在代码里，不散落在配置中

**加工具、加 Agent 都不用改核心代码**，这正是分层的目的。

---

## 五、快速开始

### 安装

```bash
pip install -e ".[dev]"
```

⚠️ **刻意不声明 Milvus / LangChain 等重型依赖**。工具层通过 `requires` 字段声明
需要哪些能力，由部署环境按需安装 —— 否则「只想看披露逻辑」的人也得装一整套检索栈。

### 跑起来

```bash
# 默认 profile，stdio 传输（给本地客户端）
python server.py

# 换 profile
python server.py --profile dev-agent

# HTTP 传输（远程多客户端）
python server.py --transport http --host 0.0.0.0 --port 8765
```

### 不开客户端就能核对暴露面

```bash
python server.py --profile health-agent --list-tools
```

输出该 profile 下**可见**的工具清单与所属 scope。这个命令在改动 profile 后
是第一个该跑的 —— 比连上客户端再看快得多。

### 工具清单

**当前注册 8 个工具**，外加 2 个元工具（`meta.search_tools` / `meta.enable_tool`）。

**具体有哪些，见 [`registry/registry.yaml`](registry/registry.yaml)** —— 不在 README
里重复维护一份，否则每加一个工具就要改两处，必然会漂移。

**按 scope 分布**（`--list-tools` 可实测）：

| scope | 工具数 | health-agent 是否可见 |
|---|:--:|:--:|
| `common.discovery` | 2 | ✅ |
| `common.knowledge` | 1 | ✅ |
| `common.safety` | 3 | ✅ |
| `common.ops.read` | 1 | ✅ |
| `common.ops.write` | 1 | ❌ **不可见** |

**⇒ health-agent 可见 7/8，dev-agent 可见 8/8。** 差的那一个是重建索引。

---

## 六、权限模型

### 四个概念

| 概念 | 是什么 | 谁定义 |
|---|---|---|
| **scope** | 权限单位（如 `common.ops.write`） | `registry.yaml` 的 `scopes` |
| **tool** | 具体能力，必须归属某个 scope | `registry.yaml` 的 `tools` |
| **profile** | 一个 Agent 的授权声明 | `profiles/*.yaml` |
| **principal** | 运行时的身份，持有已授予 scope | 由 profile 引导初始化 |

### 启动校验（8 类）

registry 加载时会做 8 类校验，任何一条不过就**启动失败**（而不是运行到一半才发现）：

- 工具未声明 scope
- scope 不在已知域
- 降级链跨权限域
- `requires` 引用了未定义的能力
- 等等

**⇒ 配置错误在启动时暴露，不在第一次调用时暴露。**

### 降级链

工具可以声明 `fallback`，指定「挂了用谁代替」。

**约束：只允许同权限域内降级。** `common.*` 只能降到 `common.*` ——
否则「降级」会变成提权的后门。

---

## 七、测试

```bash
pytest -q     # 53 passed
```

四个维度各自独立：

| 文件 | 条数 | 测什么 |
|---|:--:|---|
| `test_registry.py` | 17 | 注册表校验、scope 归属、降级链约束 |
| `test_disclosure.py` | 15 | 可见性计算、enable 提权路径、越权拒绝 |
| `test_resilience.py` | 11 | 降级与熔断 |
| `test_observability.py` | 10 | 审计与指标 |

**为什么不依赖真实 Milvus**：披露层的正确性不该由「能不能连上向量库」决定。
需要外部能力时用 `requires` 声明并在测试里打桩。

---

## 八、已知边界

- **不带传输层鉴权** —— HTTP 模式下没有 token 校验。生产部署应放在反向代理后，
  由前置层做认证
- **principal 状态存在进程内存** —— 进程重启后回到 profile 的初始授权。
  跨重启保持需要接入外部存储（接口已留，未实现）
- **不支持热更新 registry** —— 加工具要重启。工具清单变动频率低，不值得为它引入
  文件监听与并发一致性处理
- **审计日志只到工具级** —— 记录「谁调了哪个工具、成败、耗时」，不记录参数内容
  （健康领域参数可能含个人信息）

---

## 九、License

MIT