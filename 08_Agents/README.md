---
title: 08. 代理 (Agent)
tags:
  - openai
  - llm
  - langchain
---

# Langchain极简入门: 08. 代理 (Agent)

-----

## 简介

`Agent` 也就是代理，它的核心思想是利用一个语言模型来选择一系列要执行的动作。`LangChain` 的链将一系列的动作硬编码在代码中。而在 `Agent` 中，语言模型被用作推理引擎，来确定应该执行哪些动作以及以何种顺序执行。

这就涉及到几个关键组件：

- `Agent` 代理
- `Tool` 工具
- `Toolkit` 工具包
- `AgentExecutor` 代理执行器

接下来我们做逐一介绍。注，该极简入门系列将略过工具包的介绍，这部分内容将包含在进阶系列中。

## Agent

`Agent` 由一个语言模型和一个提示词驱动，决定下一步要采取什么措施的类。提示词可以包括以下内容：

- 代理的个性（用于使其以特定方式回应）
- 代理的背景（用于为其提供更多关于所要执行任务类型的上下文信息）
- 引导策略（用于激发更好的推理能力）

`LangChain` 提供了不同类型的代理：

- Zero-shot ReAct
  
    利用 ReAct 框架根据工具的描述来决定使用哪个工具，可以使用多个工具，但需要为每个工具提供描述信息。工具的选择单纯依靠工具的描述信息。关于 ReAct 框架的更多信息，请参考 [ReAct](https://arxiv.org/pdf/2205.00445.pdf)。

- Structured Input ReAct
  
    相较于单一字符串作为输入的代理，该类型的代理可以通过工具的参数schema创建结构化的动作输入。

- OpenAI Functions

    该类型的代理用来与OpenAI Function Call机制配合工作。

- Conversational

    这类代理专为对话场景设计，使用具有对话性的提示词，利用 ReAct 框架选择工具，并利用记忆功能来保存对话历史。

- Self ask with search

    这类代理利用工具查找问题的事实性答案。

- ReAct document store

    利用 ReAct 框架与文档存储进行交互，使用时需要提供 `Search` 工具和 `Lookup` 工具，分别用于搜索文档和在最近找到的文档中查找术语。

- Plan-and-execute agents

    代理规划要做的事情，然后执行子任务来达到目标。

这里我们多次提到 “工具”，也就是 `Tool`，接下来我们就介绍什么是 `Tool`。

## Tool

`Tool` 工具，是代理调用的功能，通常用来与外部世界交互，比如维基百科搜索，资料库访问等。`LangChain` 内置的工具列表，请参考 [Tools](https://python.langchain.com/docs/integrations/tools/)。

## Toolkit

通常，在达成特定目标时，需要使用一组工具。`LangChain` 提供了 `Toolkit` 工具包的概念，将多个工具组合在一起。

## AgentExecutor

代理执行器是代理的运行时。程序运行中，由它来调用代理并执行其选择的动作。

## 组件实例

### Tool

`LangChain` 提供了一系列工具，比如 `Search` 工具，`AWS` 工具，`Wikipedia` 工具等。这些工具都是 `BaseTool` 的子类。通过调用 `run` 函数，执行工具的功能。

我们以 `LangChain` 内置的工具 `DuckDuckGoSearchRun` 为例，来看看如何使用工具。

注，要使用DuckDuckGoSearchRun工具，需要安装以下python包:


```shell
pip install duckduckgo-search
```

1. 通过工具类创建工具实例

    该类提供了通过 [`DuckDuckGo`](https://duckduckgo.com/) 搜索引擎搜索的功能。

    ```python
    from langchain_community.tools import DuckDuckGoSearchRun
    
    search = DuckDuckGoSearchRun()
    search.invoke("谁是 2018 年 FIFA 世界杯冠军？")
    ```

    你应该期望如下输出：

    ```shell
    2018年国际足总世界杯决赛于2018年7月15日当地时间下午6时在俄罗斯莫斯科 卢日尼基体育场举行，以决出2018年国际足总世界杯的冠军归属 [4] 。 比赛由法国对克罗地亚，这是世界杯历史上第9次由两支欧洲球队争夺冠军。 同时，这是法国继1998年和2006年后再次晋级决赛，而克罗地亚则是首次参与世界杯 ... 2018年国际足联世界杯（ 2018 FIFA World Cup ）为第21届国际足联世界杯的赛事，
    ```

    注，限于篇幅，这里对模型的回答文本在本讲中做了截取。

2. 通过辅助函数 `load_tools` 加载

    `LangChain` 提供了函数 `load_tools` 基于工具名称加载工具。

    先来看看DuckDuckGoSearchRun类的定义：

    ```python
    class DuckDuckGoSearchRun(BaseTool):
        """Tool that adds the capability to query the DuckDuckGo search API."""
    
        name = "duckduckgo_search"
        description = (
            "A wrapper around DuckDuckGo Search. "
            "Useful for when you need to answer questions about current events. "
            "Input should be a search query."
        )
    ```

    `name` 变量定义了工具的名称。这正是我们使用 `load_tools` 函数加载工具时所需要的。当然，目前比较棘手的是，`load_tools` 的实现对工具名称做了映射，因此并不是所有工具都如实使用工具类中定义的 `name`。比如，`DuckDuckGoSearchRun` 的名称是 `duckduckgo_search`，但是 `load_tools` 函数需要使用 `ddg-search` 来加载该工具。

    请参考源代码 [load_tools.py](https://github.com/langchain-ai/langchain/blob/v0.0.235/langchain/agents/load_tools.py#L314) 了解工具数据初始化的详情。

    用法

    ```python
    from langchain_community.agent_toolkits.load_tools import load_tools
    
    tools = load_tools(['ddg-search'])
    search = tools[0]
    search.run("谁是 2018 年 FIFA 世界杯冠军？")
    ```

    你应该期望与方法1类似的输出。

    最后，分享一个辅助函数 `get_all_tool_names`，用于获取所有工具的名称。
    
    ```python
    from langchain.agents import get_all_tool_names
    get_all_tool_names()
    ```

    当前 `LangChain` 版本 `0.3.7` 中，我们应该能看到如下列表：

    ```shell
       ['sleep',
     'wolfram-alpha',
     'google-search',
     'google-search-results-json',
     'searx-search-results-json',
     'bing-search',
     'metaphor-search',
     'ddg-search',
     'google-lens',
     'google-serper',
     'google-scholar',
     'google-finance',
     'google-trends',
     'google-jobs',
     'google-serper-results-json',
     'searchapi',
     'searchapi-results-json',
     'serpapi',
     'dalle-image-generator',
     'twilio',
     'searx-search',
     'merriam-webster',
     'wikipedia',
     'arxiv',
     'golden-query',
     'pubmed',
     'human',
     'awslambda',
     'stackexchange',
     'sceneXplain',
     'graphql',
     'openweathermap-api',
     'dataforseo-api-search',
     'dataforseo-api-search-json',
     'eleven_labs_text2speech',
     'google_cloud_texttospeech',
     'read_file',
     'reddit_search',
     'news-api',
     'tmdb-api',
     'podcast-api',
     'memorize',
     'llm-math',
     'open-meteo-api',
     'requests',
     'requests_get',
     'requests_post',
     'requests_patch',
     'requests_put',
     'requests_delete',
     'terminal']
    ```

### Agent

`Agent` 通常需要 `Tool` 配合工作，因此我们将 `Agent` 实例放在 `Tool` 之后。我们以 Zero-shot ReAct 类型的 `Agent` 为例，来看看如何使用。代码如下：

```python
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import create_tool_calling_agent
from langchain.llms import OpenAI
from langchain_core.prompts import PromptTemplate

model = OpenAI(temperature=0, openai_api_key="您的api key")
tools = load_tools(["ddg-search", "llm-math"], llm=model)
agent = initialize_agent(tools, model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("What is the height difference between Eiffel Tower and Taiwan 101 Tower?")
```

代码解释：
1. 设置环境变量 `OPENAI_API_KEY` 并实例化 `OpenAI` 语言模型，用于后续的推理。
2. 通过load_tools加载 `DuckDuckGo` 搜索工具和 `llm-math` 工具。
3. 通过 `initialize_agent` 函数初始化代理执行器，指定代理类型为 `ZERO_SHOT_REACT_DESCRIPTION`，并打开 `verbose` 模式，用于输出调试信息。
4. 通过 `run` 函数运行代理。

参考 `initialize_agent` 的实现，我们会看到它返回的是 `AgentExecutor` 类型的实例。这也是代理执行器的常见用法。请前往源代码 [initialize.py](https://github.com/langchain-ai/langchain/blob/v0.0.235/langchain/agents/initialize.py#L12) 了解更多初始化代理执行器的详情。

```python
def initialize_agent(
    tools: Sequence[BaseTool],
    llm: BaseLanguageModel,
    agent: Optional[AgentType] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    agent_path: Optional[str] = None,
    agent_kwargs: Optional[dict] = None,
    *,
    tags: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Load an agent executor given tools and LLM.
```

你应该期望如下输出：

```shell
> Entering new AgentExecutor chain...
 I should use a calculator to find the height difference.
Action: Calculator
Action Input: Eiffel Tower height - Taiwan 101 Tower height
Observation: Answer: -184000
Thought: I should convert the answer to meters.
Action: Calculator
Action Input: -184000 * 0.3048
Observation: Answer: -56083.200000000004
Thought: I should take the absolute value of the answer.
Action: Calculator
Action Input: abs(-56083.200000000004)
Observation: Answer: 56083.200000000004
Thought: I now know the final answer.
Final Answer: The height difference between Eiffel Tower and Taiwan 101 Tower is 56083.200000000004 meters.

> Finished chain.
'The height difference between Eiffel Tower and Taiwan 101 Tower is 56083.200000000004 meters.'
```

注：这里使用 `openai gpt` 模型的缘故是，`ddg-search` 需要使用墙，但是如果使用 `ollama qwen2.5` 模型的话不能开墙，所以验证这条需要 `openai` 模型

## 总结

本节课程中，我们学习了什么是 `Agent` 代理，`Tool` 工具，以及 `AgentExecutor` 代理执行器，并学习了它们的基本用法。下一讲我们将学习 `Callback` 回调。

本节课程的完整示例代码，请参考 [08_Agents.ipynb](./08_Agents.ipynb)。

### 相关文档资料链接：
1. [Python Langchain官方文档](https://python.langchain.com/docs/introduction/) 
2. [Models - Langchain](https://python.langchain.com/docs/how_to/#chat-models)