---
title: 02. 模型
tags:
  - ollama
  - openai
  - llm
  - langchain
---

# Langchain极简入门: 02. 模型

-----

## 模型简介

Langchain所封装的模型分为两类：

- 大语言模型 (LLM)
- 聊天模型 (Chat Models)

在后续的内容中，为简化描述，我们将使用 `LLM` 来指代大语言模型。

Langchain的支持众多模型供应商，包括OpenAI、ChatGLM、HuggingFace等。本教程中，我们将以OpenAI为例，后续内容中提到的模型默认为OpenAI提供的模型。

Langchain的封装，比如，对OpenAI模型的封装，实际上是指的是对OpenAI API的封装。

### LLM

`LLM` 是一种基于统计的机器学习模型，用于对文本数据进行建模和生成。LLM学习和捕捉文本数据中的语言模式、语法规则和语义关系，以生成连贯并合乎语言规则的文本。

在Langchain的环境中，LLM特指文本补全模型（text completion model）。

注，文本补全模型是一种基于语言模型的机器学习模型，根据上下文的语境和语言规律，自动推断出最有可能的下一个文本补全。

| 输入         | 输出         |
| ------------ | ------------ |
| 一条文本内容 | 一条文本内容 |

### 聊天模型 (Chat Models)

聊天模型是语言模型的一种变体。聊天模型使用语言模型，并提供基于"聊天消息"的接口。

| 输入         | 输出         |
| ------------ | ------------ |
| 一组聊天消息 | 一条聊天消息 |

`聊天消息` 除了消息内容文本，还会包含一些其他参数数据。这在后续的内容中会看到。

## Langchain与OpenAI模型

参考OpenAI [Model endpoint compatibility](https://platform.openai.com/docs/models/model-endpoint-compatibility) 文档，gpt模型都归为了聊天模型，而davinci, curie, babbage, ada模型都归为了文本补全模型。

| ENDPOINT             | MODEL NAME                                                   |
| -------------------- | ------------------------------------------------------------ |
| /v1/chat/completions | gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613, gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613 |
| /v1/completions      | (Legacy)	text-davinci-003, text-davinci-002, text-davinci-001, text-curie-001, text-babbage-001, text-ada-001, davinci, curie, babbage, ada |

Langchain提供接口集成不同的模型。为了便于切换模型，Langchain将不同模型抽象为相同的接口 `BaseLanguageModel`，并提供 `predict` 和 `predict_messages` 函数来调用模型。

当使用LLM时推荐使用predict函数，当使用聊天模型时推荐使用predict_messages函数。

## 示例代码

接下来我们来看看如何在Langchain中使用LLM和聊天模型。

[Models.ipynb](./Models.ipynb)

### 与LLM的交互

与LLM的交互，我们需要使用 `langchain_ollama` 模块中的 `OllamaLLM` 类。

```python
from langchain_ollama import OllamaLLM
llm = OllamaLLM(temperature=0, model="qwen2.5:14b")
llm.invoke("什么是 AI?")
```

你应该能看到类似如下输出：

```shell
'AI，即人工智能（Artificial Intelligence），是指由人制造出来的具有一定智能的系统，它能够通过感知环境并采取行动来实现特定的目标。这些目标可以是解决问题、模式识别、语言处理等任务。\n\nAI的研究领域包括但不限于机器学习、深度学习、自然语言处理、计算机视觉和机器人技术。随着计算能力的进步以及大数据的发展，人工智能在近年来取得了显著进展，并被广泛应用于各个行业，如医疗健康、金融服务、教育、交通等领域，极大地提高了效率并创造了新的可能性。\n\n简而言之，AI是让机器能够执行通常需要人类智能才能完成的任务的技术集合。'
```

### 与聊天模型的交互

与聊天模型的交互，我们需要使用 `langchain_openai` 模块中的 `ChatOpenAI` 类。

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

chat = ChatOpenAI(temperature=0, model_name="qwen2.5:14b", openai_api_key='ollama', openai_api_base='http://localhost:11434/v1')
response = chat.invoke([
  HumanMessage(content="什么是 AI?")
])
print(response)
```

你应该能看到类似如下输出：

```shell
content='AI，即人工智能（Artificial Intelligence），是指由人制造出来的具有一定智能的系统，它能够通过感知环境并采取行动来实现特定的目标。这些目标可以是解决问题、模式识别、语言处理等任务。\n\nAI的研究领域包括但不限于机器学习、深度学习、自然语言处理、计算机视觉和机器人技术。随着计算能力的进步以及大数据的发展，人工智能在近年来取得了显著进展，并被广泛应用于各个行业，如医疗健康、金融服务、教育、交通等领域，极大地提高了效率并创造了新的可能性。\n\n简而言之，AI是让机器能够执行通常需要人类智能才能完成的任务的技术集合。' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 128, 'prompt_tokens': 36, 'total_tokens': 164, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'qwen2.5:14b', 'system_fingerprint': 'fp_ollama', 'finish_reason': 'stop', 'logprobs': None} id='run-ddd446c2-8e37-4b62-899b-3c489556a82e-0' usage_metadata={'input_tokens': 36, 'output_tokens': 128, 'total_tokens': 164, 'input_token_details': {}, 'output_token_details': {}}
```

通过以下代码我们查看一下 `response` 变量的类型：

```python
response.__class
```

可以看到，它是一个 `AIMessage` 类型的对象。

```shell
langchain_core.messages.ai.AIMessage
```

```shell
接下来我们使用 `SystemMessage` 指令来指定模型的行为。如下代码指定模型对AI一无所知，在回答AI相关问题时，回答“我不知道。”。

response = chat.predict_messages([
  SystemMessage(content="你是一个对人工智能一无所知的聊天机器人。当被问及人工智能时，你必须说 '我不知道。'"),
  HumanMessage(content="什么是深度学习？")
])
print(response)
```

你应该能看到类似如下输出：

```shell
content='我不知道。' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 3, 'prompt_tokens': 46, 'total_tokens': 49, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'qwen2.5:14b', 'system_fingerprint': 'fp_ollama', 'finish_reason': 'stop', 'logprobs': None} id='run-786599c0-7b24-4b8f-a56e-2dccaa5cee42-0' usage_metadata={'input_tokens': 46, 'output_tokens': 3, 'total_tokens': 49, 'input_token_details': {}, 'output_token_details': {}}
```

#### 3个消息类

Langchain框架提供了三个消息类，分别是 `AIMessage`、`HumanMessage` 和 `SystemMessage`。它们对应了OpenAI聊天模型API支持的不同角色 `assistant`、`user` 和 `system`。请参考 [OpenAI API文档 - Chat - Role](https://platform.openai.com/docs/api-reference/chat/create#chat/create-role)。

| Langchain类   | OpenAI角色 | 作用                         |
| ------------- | ---------- | ---------------------------- |
| AIMessage     | assistant  | 模型回答的消息               |
| HumanMessage  | user       | 用户向模型的请求或提问       |
| SystemMessage | system     | 系统指令，用于指定模型的行为 |

## 总结

本节课程中，我们学习了模型的基本概念，LLM与聊天模型的差异，并基于 `Langchain` 实现了分别与OpenAI LLM和聊天模型的交互。

要注意，虽然是聊天，但是当前我们所实现的交互并没有记忆能力，也就是说，模型并不会记住之前的对话内容。在后续的内容中，我们将学习如何实现记忆能力。

### 相关文档资料链接：

1. [Python Langchain官方文档](https://python.langchain.com/docs/introduction/) 
2. [Models - Langchain](https://python.langchain.com/docs/how_to/#chat-models)