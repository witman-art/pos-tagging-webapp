# 英文词性自动标注系统（POS Tagging Web App）

基于 **BiLSTM-CRF** 的英文词性自动标注系统，使用 Flask 封装为本地 Web 应用。用户可通过浏览器输入英文文本或上传 TXT/DOCX 文件，实时获得词性标注结果，并导出为 HTML / CSV / Excel 格式。

在 Penn Treebank WSJ 测试集上 Token Accuracy 达到 **93.45%**。

## 功能特性

- 浏览器内英文文本输入与实时词性标注
- 支持 TXT / DOCX 文件批量导入解析
- 标注结果可视化展示，支持导出为 HTML / CSV / Excel
- 模型实例缓存机制，避免重复加载，显著降低二次请求响应时间
- 完善的异常处理：空文本、错误文件格式、依赖资源缺失等边界情况

## 技术栈

- **后端**：Python、Flask、NLTK
- **模型**：PyTorch、pytorch-crf（BiLSTM + CRF）
- **前端**：HTML / CSS / JavaScript
- **文件处理**：python-docx、openpyxl
- **可视化**：matplotlib（训练曲线）

## 项目结构

```
.
├── app.py                # Flask 主程序，定义路由与接口
├── model.py              # BiLSTM-CRF 模型定义
├── predictor.py          # 模型加载与推理封装
├── data_utils.py         # 数据预处理与词表构建
├── nlp_fixes.py          # NLTK 资源处理与备用分词逻辑
├── train.py              # 模型训练入口
├── comparison_model.py   # 基线模型对比（BiLSTM + CE）
├── system_test.py         # 系统功能测试脚本
├── templates/            # 前端页面
├── static/               # 样式与静态资源
└── checkpoints/          # 模型权重（不纳入版本管理）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载 NLTK 数据

首次运行需下载 NLTK 分词与词性标注相关资源：

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

### 3. 启动服务

```bash
python app.py
```

浏览器访问 <http://127.0.0.1:5000>

### 4. 运行测试

```bash
python system_test.py
```

覆盖模型状态检测、文本标注、空文本异常处理、文件导入导出、错误格式处理等场景。

## 模型说明

- **BiLSTM 层**：提取单词的双向上下文特征
- **CRF 层**：通过标签转移约束，输出整句最优标签序列，避免不合理的词性组合
- **训练数据**：Penn Treebank WSJ 语料库
- **测试指标**：Token Accuracy 93.45%（WSJ test set）

## 作者

witman-art — 个人设计
