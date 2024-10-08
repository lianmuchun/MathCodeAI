# MathCode AI

## 概述

**MathCode AI** 是一个基于人工智能的Web应用程序，旨在帮助用户与文档进行智能交互。它处理PDF文件，并通过问答功能帮助用户提取有用的信息。该应用程序结合了Streamlit用于前端界面，Ollama用于语言模型，和LangChain用于文档处理与检索。

## 功能

- **PDF上传与处理**：用户可以上传PDF文件，系统会将其处理为可搜索的向量数据库。
- **交互式问答**：用户可以就PDF内容提出问题，MathCode AI会根据文档内容生成最相关的回答。
- **模型选择**：用户可以从可用的AI模型列表中选择一个，以适应不同的需求。
- **会话管理**：应用程序允许用户删除向量数据库并重置会话状态。

## 安装

### 系统要求

请确保在使用前已在本地安装 [Ollama](https://ollama.com)。Ollama用于支持应用程序中的语言模型处理功能。

此外，请确保已安装Python，并通过 `requirements.txt` 文件安装所需的依赖库：

```bash
pip install -r requirements.txt
```

主要依赖库包括：

- `streamlit==1.37.1`
- `ollama==0.3.2`
- `langchain==0.2.15`
- `langchain_community==0.2.14`
- `langchain_core==0.2.36`
- `langchain_text_splitters==0.2.2`
- `pdfplumber==0.11.4`

### 运行应用程序

要运行MathCode AI应用程序，使用以下命令：

```bash
streamlit run mathcodeai.py
```

此命令将启动Streamlit服务器，并在您的Web浏览器中打开应用程序。

## 使用说明

1. **上传PDF**：通过上传器上传一个PDF文档。
2. **选择模型**：从下拉列表中选择一个AI模型，用于处理您的问题。
3. **提问**：在输入框中输入您的问题。系统会根据PDF内容处理您的问题并生成回答。
4. **管理会话**：使用“删除会话”按钮可以清除当前会话并删除临时文件。

## 测试文件

在 `聯營PI01（已切分）.pdf` 文件中，包含了测试MathCode AI功能的示例数据。该文件可以用于演示和测试程序的PDF处理与问答功能。

## 日志记录

应用程序会记录重要的活动和错误，以便调试和监控。日志配置为显示 `INFO` 级别及以上的消息。

## 贡献者

MathCode AI项目是以下团队成员的共同努力成果：

- 李智琛
- 刘红斌
- 卢兴湛
- 曹俊泽
- 周济坤
- 梁思杰
- 周洁
