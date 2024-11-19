---
title: 09. 回调 (Callback)
tags:
  - openai
  - ollama
  - llm
  - langchain
---

# Langchain极简入门: 09. 回调 (Callback)

-----

## 简介

`Callback` 是 `LangChain` 提供的回调机制，允许我们在 `LLM` 应用程序的各个阶段使用 `Hook`（钩子）。这对于记录日志、监控、流式传输等任务非常有用。这些任务的执行逻辑由回调处理器（`CallbackHandler`）定义。

在 `Python` 程序中， 回调处理器通过继承 `BaseCallbackHandler` 来实现。`BaseCallbackHandler` 接口对每一个可订阅的事件定义了一个回调函数。`BaseCallbackHandler` 的子类可以实现这些回调函数来处理事件。当事件触发时，`LangChain` 的回调管理器 `CallbackManager` 会调用相应的回调函数。

以下是 `BaseCallbackHandler` 的定义。请参考[源代码](https://github.com/langchain-ai/langchain/blob/v0.0.235/langchain/callbacks/base.py#L225)。

```python
class BaseCallbackHandler:
    """Base callback handler that can be used to handle callbacks from langchain."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
```

`LangChain` 内置支持了一系列回调处理器，我们也可以按需求自定义处理器，以实现特定的业务。

## 内置处理器

`StdOutCallbackHandler` 是 `LangChain` 所支持的最基本的处理器。它将所有的回调信息打印到标准输出。这对于调试非常有用。

`LangChain` 链的基类 `Chain` 提供了一个 `callbacks` 参数来指定要使用的回调处理器。请参考[`Chain源码`](https://github.com/langchain-ai/langchain/blob/v0.0.235/langchain/chains/base.py#L63)，其中代码片段为：

```python
class Chain(Serializable, ABC):
    """Abstract base class for creating structured sequences of calls to components.
    ...
    callbacks: Callbacks = Field(default=None, exclude=True)
    """Optional list of callback handlers (or callback manager). Defaults to None.
    Callback handlers are called throughout the lifecycle of a call to a chain,
    starting with on_chain_start, ending with on_chain_end or on_chain_error.
    Each custom chain can optionally call additional callback methods, see Callback docs
    for full details."""
```

用法如下：

```python
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

handler = StdOutCallbackHandler()
llm = OllamaLLM(temperature=0, model="qwen2.5:14b", callbacks=[handler])
template = "Who is {name}?"
prompt = PromptTemplate(template=template, input_variables=['name'])
chain = prompt | llm | StrOutputParser()
chain.invoke("Super Mario")
```

你应该期望如下输出：

```shell
'Super Mario is one of the most iconic video game characters in history, created by Nintendo. He first appeared in 1985 as the protagonist of "Super Mario Bros." for the Nintendo Entertainment System (NES). The character was originally named "Jumpman" but was later renamed to Mario after the Italian plumber Mr. Video Game designer Shigeru Miyamoto chose the name.\n\nMario is typically depicted wearing a red hat and overalls with blue suspenders, and he has brown hair and a bushy moustache. He\'s known for his ability to jump very high and far, which helps him rescue Princess Peach from the clutches of Bowser, the main antagonist in many Mario games. Over time, Super Mario has appeared in numerous video games across various Nintendo platforms, including handheld devices like the Game Boy and more recent consoles such as the Switch.\n\nIn addition to platformers, Mario also stars in other game genres, such as racing (Mario Kart series), sports (Mario Tennis, Mario Golf), puzzle-solving (Super Mario 3D World), and role-playing games. The character has become a cultural icon and is one of Nintendo\'s most recognizable mascots worldwide.'
```

## 自定义处理器

我们可以通过继承 `BaseCallbackHandler` 来实现自定义的回调处理器。下面是一个简单的例子，`TimerHandler` 将跟踪 `Chain` 或 `LLM` 交互的起止时间，并统计每次交互的处理耗时。

```python
from langchain_core.callbacks.base import BaseCallbackHandler
import time

class TimerHandler(BaseCallbackHandler):

    def __init__(self) -> None:
        super().__init__()
        self.previous_ms = None
        self.durations = []

    def current_ms(self):
        return int(time.time() * 1000 + time.perf_counter() % 1 * 1000)

    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        self.previous_ms = self.current_ms()

    def on_chain_end(self, outputs, **kwargs) -> None:
        if self.previous_ms:
          duration = self.current_ms() - self.previous_ms
          self.durations.append(duration)

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self.previous_ms = self.current_ms()

    def on_llm_end(self, response, **kwargs) -> None:
        if self.previous_ms:
          duration = self.current_ms() - self.previous_ms
          self.durations.append(duration)

llm = OllamaLLM(temperature=0, model="qwen2.5:14b", callbacks=[handler])
timerHandler = TimerHandler()
prompt = PromptTemplate.from_template("What is the HEX code of color {color_name}?")
chain = prompt | llm
response = chain.invoke(input="blue")
print(response)
response = chain.invoke(input="purple")
print(response)

timerHandler.durations
```

你应该期望如下输出：

```shell
 The HEX code for the color blue can vary depending on which specific shade of blue you're referring to, but one common representation of blue in web design and digital contexts is #0000FF or a lighter variant like #ADD8E6. If you are looking for a particular shade of blue, please specify!
 The HEX code for the color purple can vary depending on which specific shade of purple you're referring to, but one common representation of purple in web design and digital contexts is #800080. This is a medium shade of purple often referred to as "purple" or "web purple." If you need a different shade of purple, please specify the tone or provide more details!
[1589, 1097]
```

## 回调处理器的适用场景

通过 `LLMChain` 的构造函数参数设置 `callbacks` 仅仅是众多适用场景之一。接下来我们简明地列出其他使用场景和示例代码。

对于 `Model`，`Agent`， `Tool`，以及 `Chain` 都可以通过以下方式设置回调处理器：
### 1. 构造函数参数 `callbacks` 设置

关于 `Chain`，以 `LLMChain` 为例，请参考本讲上一部分内容。注意在 `Chain` 上的回调器监听的是 `chain` 相关的事件，因此回调器的如下函数会被调用：
- on_chain_start
- on_chain_end
- on_chain_error

`Agent`， `Tool`，以及 `Chain` 上的回调器会分别被调用相应的回调函数。

下面分享关于 `Model` 与 `callbacks` 的使用示例：

```python
timerHandler = TimerHandler()
llm = OllamaLLM(temperature=0, model="qwen2.5:14b", callbacks=[handler])
response = llm.invoke("What is the HEX code of color BLACK?")
print(response)

timerHandler.durations
```

你应该期望看到类似如下的输出：

```shell
['What is the HEX code of color BLACK?']
generations=[[Generation(text='\n\nThe hex code of black is #000000.', generation_info={'finish_reason': 'stop', 'logprobs': None})]] llm_output={'token_usage': {'prompt_tokens': 10, 'total_tokens': 21, 'completion_tokens': 11}, 'model_name': 'text-davinci-003'} run=None


The hex code of black is #000000.

[1223]
```

### 2. 通过运行时的函数调用

`Model`，`Agent`， `Tool`，以及 `Chain` 的请求执行函数都接受 `callbacks` 参数，比如 `LLMChain` 的 `run` 函数，`OpenAI` 的 `predict` 函数，等都能接受 `callbacks` 参数，在运行时指定回调处理器。

以 `OpenAI` 模型类为例：

```python
timerHandler = TimerHandler()
llm = OllamaLLM(temperature=0, model="qwen2.5:14b", callbacks=[handler])
response = llm.invoke("What is the HEX code of color BLACK?", callbacks=[timerHandler])
print(response)

timerHandler.durations
```

你应该同样期望如下输出：

```shell
['What is the HEX code of color BLACK?']
generations=[[Generation(text='\n\nThe hex code of black is #000000.', generation_info={'finish_reason': 'stop', 'logprobs': None})]] llm_output={'token_usage': {'prompt_tokens': 10, 'total_tokens': 21, 'completion_tokens': 11}, 'model_name': 'text-davinci-003'} run=None

The hex code of black is #000000.

[1138]
```

关于 `Agent`，`Tool` 等的使用，请参考官方文档API。

## 总结

本节课程中，我们学习了什么是 `Callback` 回调，如何使用回调处理器，以及在哪些场景下可以接入回调处理器。下一讲，我们将一起完成一个完整的应用案例，来巩固本系列课程的知识点。

本节课程的完整示例代码，请参考 [09_Callbacks.ipynb](./09_Callbacks.ipynb)。

### 相关文档资料链接：
1. [Python Langchain官方文档](https://python.langchain.com/docs/introduction/) 
2. [Models - Langchain](https://python.langchain.com/docs/how_to/#chat-models)