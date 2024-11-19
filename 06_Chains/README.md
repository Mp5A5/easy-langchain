---
title: 06. 链
tags:
  - openai
  - ollama
  - llm
  - langchain
---

# Langchain极简入门: 06. 链

-----

## 简介

单一的LLM对于简单的应用场景已经足够，但是更复杂的应用程序需要将LLM串联在一起，需要多LLM协同工作。

LangChain提出了 `链` 的概念，为这种“链式”应用程序提供了 **Chain** 接口。`Chain` 定义组件的调用序列，其中可以包括其他链。链大大简化复杂应用程序的实现，并使其模块化，这也使调试、维护和改进应用程序变得更容易。

## 最基础的链 LLMChain

作为极简教程，我们从最基础的概念，与组件开始。`LLMChain` 是 `LangChain` 中最基础的链。本课就从 `LLMChain` 开始，学习链的使用。

`LLMChain` 接受如下组件：
- LLM
- 提示词模版

`LLMChain` 返回LLM的回复。

在[第二讲](../02_Models/README.md)中我们学习了OpenAI LLM的使用。现在我们基于OpenAI LLM，利用 `LLMChain` 尝试构建自己第一个链。

1. 准备必要的组件

```python
llm = OllamaLLM(temperature=0, model="qwen2.5:14b")
prompt = PromptTemplate(
    input_variables=["color"],
    template="{color}的十六进制代码是什么？",
)
```

2. 基于组件创建 `LLMChain` 实例

我们要创建的链，基于提示词模版，提供基于颜色名字询问对应的16进制代码的能力。

```python
chain = prompt | llm | StrOutputParser()
```

3. 基于链提问

现在我们利用创建的 `LLMChain` 实例提问。注意，提问中我们只需要提供第一步中创建的提示词模版变量的值。我们分别提问green，cyan，magento三种颜色的16进制代码。

```python
print(chain.run("green"))
print(chain.run("cyan"))
print(chain.run("magento"))
```

你应该期望如下输出：

```shell
看起来像是没有直接给出答案。实际上，常见的绿色在网页设计和计算机科学中的十六进制代码是 #008000（纯正绿色），不过也有许多其他不同的绿色色调。如果你指的是标准的颜色名称“Green”或“Lime”，它们的十六进制代码分别是：

- 绿色 (Green): #008000
- 柠檬绿 (Lime): #32CD32

当然，还有无数种其他的绿色变体，每一种都有其独特的十六进制代码。如果你有特定类型的绿色在寻找，请提供更多的细节以便给出更准确的答案。
 青色在不同的文化和语境中有不同的含义，但通常所说的“青”在中国传统色彩中指的是介于蓝和绿之间的颜色。如果要指定一个接近中国传统意义上的“青”的十六进制颜色代码，可以使用 #48D1CC 这个值，它是一种偏蓝色调的青色。不过需要注意的是，“青”这个概念在不同的语境下可能对应不同的具体颜色，因此也有其他版本的颜色代码可以代表“青”，例如更偏向绿色的青色可以用 #7FFF00 表示。
 

您的问题似乎没有提供足够的信息来确定“张三”的十六进制代码。通常，十六进制代码用于表示计算机中的数据或内存地址，并且与特定的人名无关。如果您是指某个人的名字或者身份标识符的十六进制形式，请提供更多上下文信息以便我能更好地帮助您。例如，如果这是一个用户名、身份证号或者其他具体的信息，请提供详细情况。
```

## LangChain Hub

[LangChain Hub](https://smith.langchain.com/hub) 收集并分享用于处理 `LangChain` 基本元素（提示词，链，和代理等）。

本讲，我们介绍 `LangChainHub` 中分享的链的使用。

### 从LangChainHub加载链

本课以链[math_reasoning_prompt](https://smith.langchain.com/hub/demiurg/math_reasoning_prompt)为例，介绍如何从 `LangChain Hub` 加载链并使用它。这是一个使用LLM和Python REPL来解决复杂数学问题的链。

#### 加载

使用 `load_chain` 函数从hub加载。

```python
prompt = hub.pull("demiurg/math_reasoning_prompt")
```

#### 提问

现在我们可以基于这个链提问。

```python
llm = ChatOpenAI(temperature=0, model_name="qwen2.5:14b", openai_api_key='ollama', openai_api_base='http://localhost:11434/v1')
chain = prompt | llm | StrOutputParser()
chain.invoke("whats the area of a circle with radius 2?")
```

你应该期望如下输出：

```shell
- The question asks for the area of a circle given its radius, so let's solve it step-by-step.

- Step 1: Recall the formula for calculating the area of a circle.
    - Area = π * r^2, where r is the radius of the circle and π (pi) is approximately 3.14159.
- Step 2: Substitute the given value into the formula.
    - Here, the radius (r) = 2 units.
    - Therefore, Area = π * (2)^2
- Step 3: Calculate the area.
    - Area = π * 4
    - Using π ≈ 3.14159, we get:
        - Area ≈ 3.14159 * 4
        - Area ≈ 12.56636 square units

- Final Answer: The area of the circle with radius 2 is approximately 12.57 square units (rounded to two decimal places).
```

## 总结
本节课程中，我们学习了`LangChain` 提出的最重要的概念 - 链（`Chain`） ，介绍了如何使用链，并分享了如何利用开源社区的力量 - 从 `LangChain Hub` 加载链，让LLM开发变得更加轻松。

### 相关文档资料链接：
1. [Python Langchain官方文档](https://python.langchain.com/docs/introduction/) 
2. [Models - Langchain](https://python.langchain.com/docs/how_to/#chat-models)