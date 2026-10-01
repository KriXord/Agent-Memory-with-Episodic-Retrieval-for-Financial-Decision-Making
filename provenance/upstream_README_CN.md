

# QuantAgents

一个复杂的多智能体交易分析系统，结合了技术指标、模式识别和趋势分析，使用 LangChain 和 LangGraph。该系统提供网络界面和程序化访问，用于全面的市场分析。

> QuantAgent 是一个研究导向的工具，旨在探索金融环境中的算法决策制定。其性能取决于多种变量，包括所选的语言模型、参数调优、数据完整性、市场波动性和其他随机因素。QuantAgent 生成的结果本质上是实验性的，不应被解释为实际交易或投资活动的建议。

## 🚀 功能特性

- **多智能体分析**：四个专业智能体协同工作：
  
  - **指标智能体**：计算技术指标（MACD、RSI、随机指标等）
  ![指标智能体](assets/indicator.png)
  - **模式智能体**：识别蜡烛图模式并生成可视化图表
  ![模式智能体](assets/pattern.png)
  - **趋势智能体**：分析市场趋势并生成趋势可视化
  ![趋势智能体](assets/trend.png)
  - **风险智能体**：汇总指标、模式和趋势报告，量化潜在回撤，推荐仓位大小，并定义止损阈值，实现全面的风险管理。
  ![风险智能体](assets/risk.png)
  - **决策智能体**：整合指标、模式、趋势和风险报告，发布最终交易指令——指定做多/做空信号和理由。
  ![决策智能体](assets/decision.png)

- **网络界面**：基于 Flask 的现代网络应用程序，具有：
  - 来自雅虎财经的实时市场数据
  - 交互式资产选择（股票、加密货币、商品、指数）
  - 多时间框架分析（1分钟到1天）
  - 动态图表生成
  - API 密钥管理

## 🔧 实现细节

我们使用 LangGraph 构建 QuantAgents 以确保灵活性和模块化。我们使用 gpt-4o 和 gpt-4o-mini 作为我们的深度思考和快速思考 LLM 进行实验。但是，出于测试目的，我们建议您使用 gpt-4o-mini 来节省成本，因为我们的框架会进行大量 API 调用。

**重要说明**：我们的模型需要一个可以接受图像输入的 LLM，因为我们的智能体会生成和分析视觉图表以进行模式识别和趋势分析。

### Python 使用

要在代码中使用 QuantAgents，您可以导入 trading_graph 模块并初始化 TradingGraph() 对象。.invoke() 函数将返回全面的分析。您可以运行 web_interface.py，这里也有一个快速示例：

```python
from trading_graph import TradingGraph

# 初始化交易图
trading_graph = TradingGraph()

# 使用您的数据创建初始状态
initial_state = {
    "kline_data": your_dataframe_dict,
    "analysis_results": None,
    "messages": [],
    "time_frame": "4hour",
    "stock_name": "BTC"
}

# 运行分析
final_state = trading_graph.graph.invoke(initial_state)

# 访问结果
print(final_state.get("final_trade_decision"))
print(final_state.get("indicator_report"))
print(final_state.get("pattern_report"))
print(final_state.get("trend_report"))
```

您还可以调整默认配置以设置您自己的 LLM 选择、分析参数等。

```python
from trading_graph import TradingGraph
from default_config import DEFAULT_CONFIG

# 创建自定义配置
config = DEFAULT_CONFIG.copy()
config["agent_llm_model"] = "gpt-4o-mini"  # 为智能体使用不同的模型
config["graph_llm_model"] = "gpt-4o"       # 为图逻辑使用不同的模型
config["agent_llm_temperature"] = 0.2      # 调整智能体的创造力水平
config["graph_llm_temperature"] = 0.1      # 调整图逻辑的创造力水平

# 使用自定义配置初始化
trading_graph = TradingGraph(config=config)

# 使用自定义配置运行分析
final_state = trading_graph.graph.invoke(initial_state)
```

对于实时数据，我们建议使用网络界面，因为它通过 yfinance 提供对实时市场数据的访问。系统会自动获取最近 30 个蜡烛图以获得最佳的 LLM 分析准确性。

### 配置选项

系统支持以下配置参数：

- `agent_llm_model`：单个智能体的模型（默认："gpt-4o-mini"）
- `graph_llm_model`：图逻辑和决策制定的模型（默认："gpt-4o"）
- `agent_llm_temperature`：智能体响应的温度（默认：0.1）
- `graph_llm_temperature`：图逻辑的温度（默认：0.1）

**注意**：系统使用默认的令牌限制进行综合分析。不应用人工令牌限制。

您可以在 `default_config.py` 中查看完整的配置列表。

## 📊 基准测试

`benchmark/` 文件夹包含用于在多个资产上测试 QuantAgents 系统的评估数据集。对于每个资产，我们通过 yfinance 等公共交易 API 收集 5000 个历史柱状图。

从这些数据中，我们为每个资产随机采样 100 个评估段。每个段由 100 个连续的蜡烛图序列组成，最后三个蜡烛图被保留在输入之外，以防止在测试时提示中暴露已验证的市场结果。系统在零样本设置中运行——无需任何监督微调——通过生成结构化交易报告，包括方向性决策（做多或做空）、简洁的文本理由和预测的风险回报比。

### 可用资产
- **BTC**：比特币（100 个 CSV 文件）
- **CL**：原油（100 个 CSV 文件）
- **DJI**：道琼斯工业平均指数（100 个 CSV 文件）
- **ES**：E-mini 标普 500（100 个 CSV 文件）
- **GC**：黄金期货（100 个 CSV 文件）
- **NQ**：纳斯达克期货（100 个 CSV 文件）
- **QQQ**：Invesco QQQ 信托（100 个 CSV 文件）
- **SPX**：标普 500（100 个 CSV 文件）

每个 CSV 文件包含 4 小时蜡烛图数据，具有 OHLCV（开盘、最高、最低、收盘、成交量）信息，用于评估和回测目的。

## 🛠️ 先决条件

- Python 3.10（强烈推荐用于兼容性）
- Conda（推荐）或 pip
- OpenAI API 密钥
- TA-Lib 库

## 📦 安装

### 1. 创建并激活 Conda 环境

```bash
conda create -n quantagents python=3.10
conda activate quantagents
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

如果您遇到 TA-lib-python 的问题，请尝试：

```bash
conda install -c conda-forge ta-lib
```

或访问 [TA-Lib Python 仓库](https://github.com/ta-lib/ta-lib-python) 获取详细的安装说明。

### 3. 设置 OpenAI API 密钥
您可以在我们的网络界面中稍后设置它，
![API 密钥设置](assets/apibox.png)

或将其设置为环境变量：
```bash
export OPENAI_API_KEY="your_api_key_here"
```

## 🚀 使用

### 启动网络界面

```bash
python web_interface.py
```

网络应用程序将在 `http://127.0.0.1:5000` 可用

### 网络界面功能

1. **资产选择**：从可用的股票、加密货币、商品和指数中选择
2. **时间框架选择**：分析从 1 分钟到每日间隔的数据
3. **日期范围**：为分析选择自定义日期范围
4. **实时分析**：获得带有可视化的全面技术分析
5. **API 密钥管理**：通过界面更新您的 OpenAI API 密钥

## 📺 演示

![快速预览](assets/demo.gif)

## 🤝 贡献

1. Fork 仓库
2. 创建功能分支
3. 进行更改
4. 如果适用，添加测试
5. 提交拉取请求

## 📄 许可证

本项目采用 MIT 许可证 - 详情请参阅 LICENSE 文件。

## 🙏 致谢

此仓库基于 [**LangGraph**](https://github.com/langchain-ai/langgraph)、[**OpenAI**](https://github.com/openai/openai-python)、[**yfinance**](https://github.com/ranaroussi/yfinance)、[**Flask**](https://github.com/pallets/flask) 和 [**TechnicalAnalysisAutomation**](https://github.com/neurotrader888/TechnicalAnalysisAutomation/tree/main) 构建。

## ⚠️ 免责声明

此软件仅供教育和研究目的使用。它不旨在提供财务建议。在做出投资决策之前，请始终进行自己的研究并考虑咨询财务顾问。

## 🐛 故障排除

### 常见问题

1. **TA-Lib 安装**：如果您遇到 TA-Lib 安装问题，请参考[官方仓库](https://github.com/ta-lib/ta-lib-python)获取平台特定的说明。

2. **OpenAI API 密钥**：确保您的 API 密钥在环境中或通过网络界面正确设置。

3. **数据获取**：系统使用雅虎财经获取数据。某些符号可能不可用或历史数据有限。

4. **内存问题**：对于大型数据集，考虑减少分析窗口或使用较小的时间框架。

### 支持

如果您遇到任何问题，请：
1. 检查上面的故障排除部分
2. 查看控制台中的错误消息
3. 确保所有依赖项都正确安装
4. 验证您的 OpenAI API 密钥有效且有足够的积分

## 📧 联系

如有问题、反馈或合作机会，请联系：

**邮箱**：[chenyu.you@stonybrook.edu](mailto:chenyu.you@stonybrook.edu) 