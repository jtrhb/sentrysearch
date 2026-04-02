# Video Quality Specification

SentrySearch 视频质量评估体系，用于 AI UGC 视频资产的自动化质检和打分。

---

## 一、多维打分 (Scorer)

对视频进行宏观质量评估，输出综合分和各维度子分。

### 评分维度

| 维度 | 分数范围 | 说明 |
|---|---|---|
| 人物一致性 `character_consistency` | 0-100 | 角色外观、比例、动作在视频中是否前后一致 |
| 场景一致性 `scene_consistency` | 0-100 | 背景环境、光照、透视是否稳定 |
| AI 味检测 `ai_score` | 0-100 | 视频呈现出的 AI 生成感（0=完全自然，100=明显 AI） |
| 资产相似度 `max_similarity` | 0-1 | 与已入库视频的最高余弦相似度 |

### 综合分计算

综合分 = 一致性均分 × W1 + (100 - AI分) × W2 + (1 - 相似度) × 100 × W3

默认权重：

| 权重 | 值 | 说明 |
|---|---|---|
| W1 (consistency) | 0.35 | 一致性越高越好 |
| W2 (ai) | 0.40 | AI 味越低越好 |
| W3 (similarity) | 0.25 | 越原创越好 |

> 当索引库为空（无已有资产可对比）时，相似度权重自动重新分配到一致性和 AI 味上。

### API

```
POST /score
{
  "r2_keys": ["ugc/video_001.mp4"],
  "check_similarity": true,
  "weights": {"consistency": 0.35, "ai": 0.40, "similarity": 0.25}  // 可选
}

GET /scores?min_overall=70&max_ai_score=40&limit=50
GET /scores/{source_file}
```

---

## 二、质量挑刺 (Criticizer)

对视频进行微观缺陷检测，逐一列出具体问题，标注严重度和时间戳。

### 缺陷分类

#### 1. 时序/运动 (temporal) — 4 种

| 缺陷类型 | key | 检测内容 |
|---|---|---|
| 动作不连贯 | `motion_discontinuity` | 突然跳帧、冻结、不自然的加速/减速 |
| 物理违规 | `physics_violation` | 物体反重力、不可能轨迹、不自然碰撞、悬浮 |
| 闪烁抖动 | `flickering` | 帧间时序不稳定、物体忽隐忽现 |
| 运动模糊异常 | `motion_blur_artifacts` | 不自然或缺失的运动模糊 |

#### 2. 视觉伪影 (visual) — 6 种

| 缺陷类型 | key | 检测内容 |
|---|---|---|
| 手部变形 | `hand_deformation` | 手指数量错误、手指融合、不自然手势 |
| 面部扭曲 | `facial_distortion` | 五官漂移、不对称、帧间特征位移 |
| 纹理游走 | `texture_swimming` | 表面纹理滑动、图案变形、细节不稳 |
| 边缘撕裂 | `edge_artifacts` | 物体边界模糊、锯齿、光晕 |
| 分辨率不均 | `resolution_inconsistency` | 局部清晰局部模糊/像素化 |
| 文字乱码 | `text_corruption` | 画面中文字/符号/标志不可读 |

#### 3. 角色/物体 (character) — 4 种

| 缺陷类型 | key | 检测内容 |
|---|---|---|
| 穿模 | `clipping` | 身体部位穿过其他身体或物体 |
| 比例失调 | `proportion_error` | 物体/人物/肢体相对大小错误 |
| 外观漂变 | `appearance_shift` | 角色面容/发型/特征帧间变化 |
| 服饰异常 | `clothing_anomaly` | 衣服消失/出现、花纹突变、布料物理异常 |

#### 4. 音画同步 (audio) — 4 种

| 缺陷类型 | key | 检测内容 |
|---|---|---|
| 唇形不同步 | `lip_sync` | 嘴型和语音时序/音素不匹配 |
| 音画不同步 | `audio_video_sync` | 音频和画面整体时间偏移 |
| 环境音不匹配 | `ambient_mismatch` | 环境声与视觉场景不一致 |
| 音频瑕疵 | `audio_artifacts` | 噪声、失真、爆音、不自然剪切、重复模式 |

#### 5. 构图/美学 (composition) — 4 种

| 缺陷类型 | key | 检测内容 |
|---|---|---|
| 构图问题 | `framing` | 主体截断、画面失衡、视觉重心偏移 |
| 色彩异常 | `color_anomaly` | 色带(banding)、过饱和、色调偏移 |
| 光照矛盾 | `lighting_contradiction` | 阴影方向不一致、光源互相矛盾 |
| 景深异常 | `depth_of_field` | 焦平面不一致、不自然虚化、对焦突变 |

#### 6. 内容连贯 (coherence) — 3 种

| 缺陷类型 | key | 检测内容 |
|---|---|---|
| 空间矛盾 | `spatial_impossibility` | 不可能的空间布局、透视矛盾 |
| 风格混搭 | `style_inconsistency` | 同一视频内混合不同视觉/艺术风格 |
| 逻辑错误 | `logical_error` | 事件因果不通、物理上不可能的情节 |

> 共 **6 大类 25 种缺陷类型**。无音轨的视频自动跳过 audio 类别。

### 严重度等级

每个检出缺陷标注一个严重度等级：

| 级别 | 含义 | 扣分权重 |
|---|---|---|
| `critical` | 不可接受，一眼可见的严重问题 | -25 分/个 |
| `major` | 清晰可辨的明显问题 | -12 分/个 |
| `minor` | 仔细看才能发现的小问题 | -4 分/个 |
| `nitpick` | 极微小，几乎不影响观感 | -1 分/个 |

### 质量等级

基于扣分计算 `grade_score = max(0, 100 - 总扣分)`，映射到等级：

| 等级 | 分数区间 | 含义 | 建议 |
|---|---|---|---|
| **A** | 90-100 | 优秀 | 可直接发布 |
| **B** | 75-89 | 良好 | 可发布，建议优化 |
| **C** | 60-74 | 可接受 | 需修复 major 问题后发布 |
| **D** | 40-59 | 较差 | 需大幅修改 |
| **F** | 0-39 | 不可用 | 建议重新生成 |

### 缺陷输出格式

每个缺陷包含以下字段：

```json
{
  "category": "character",
  "type": "clipping",
  "severity": "major",
  "description": "左手穿过桌面边缘",
  "timestamp": "3s-5s"
}
```

- `timestamp` 为近似时间范围，持续性问题标注 `"throughout"`

### API

```
POST /critique
{
  "r2_keys": ["ugc/video_001.mp4"]
}

GET /critiques?max_grade=C&limit=50
GET /critiques/{source_file}
```

---

## 三、技术说明

### 模型选择

| 用途 | 模型 | 说明 |
|---|---|---|
| 向量嵌入 | `gemini-embedding-2-preview` | 3072 维，用于语义搜索和相似度计算 |
| 打分评估 | `gemini-2.5-flash` | 多模态分析，一致性和 AI 味评估 |
| 质量挑刺 | `gemini-2.5-flash` | 多模态分析，逐帧缺陷检测 |

### 视频输入限制

| 限制 | 值 |
|---|---|
| 嵌入最大时长 | 120 秒 |
| 帧采样 | ≤32s: 1fps；>32s: 均匀采样至 32 帧 |
| 支持格式 | MP4, MOV (H264/H265/AV1/VP9) |
| 音频 | 嵌入不处理音频；打分和挑刺会分析音频 |

### 数据存储

| 表 | 用途 |
|---|---|
| `chunks` | 视频向量索引（embedding + 元数据） |
| `video_scores` | 多维打分结果 |
| `video_critiques` | 质量挑刺结果（缺陷列表 + 等级） |
