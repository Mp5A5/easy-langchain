---
title: 05. 输出解析器
tags:
  - openai
  - ollama
  - llm
  - langchain
---

# Langchain极简入门: 05. 输出解析器

-----

## 简介

LLM的输出为文本，但在程序中除了显示文本，可能希望获得更结构化的数据。这就是输出解析器（Output Parsers）的用武之地。

`LangChain` 为输出解析器提供了基础类 `BaseOutputParser`。不同的输出解析器都继承自该类。它们需要实现以下两个函数：

- **get_format_instructions**：返回指令指定LLM的输出该如何格式化，该函数在实现类中必须重写。基类中的函数实现如下：
```python
def get_format_instructions(self) -> str:
    """Instructions on how the LLM output should be formatted."""
    raise NotImplementedError
```
- **parse**：解析LLM的输出文本为特定的结构。函数签名如下：
```python
def parse(self, text: str) -> T
```

`BaseOutputParser` 还提供了如下函数供重载：
**parse_with_prompt**：基于提示词上下文解析LLM的输出文本为特定结构。该函数在基类中的实现为：
```python
def parse_with_prompt(self, completion: str, prompt: PromptValue) -> Any:
    """Parse the output of an LLM call with the input prompt for context."""
    return self.parse(completion)
```

## LangChain支持的输出解析器

LangChain框架提供了一系列解析器实现来满足应用在不同功能场景中的需求。它们包括且不局限于如下解析器：
- List parser
- Datetime parser
- Enum parser
- Auto-fixing parser
- Pydantic parser
- Retry parser
- Structured output parser

本讲介绍具有代表性的两款解析器的使用。

### List Parser

List Parser将逗号分隔的文本解析为列表。

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
output_parser.parse("black, yellow, red, green, white, blue")
```

你应该能看到如下输出：

```shell
['black', 'yellow', 'red', 'green', 'white', 'blue']
```

### Structured Output Parser

当我们想要类似JSON数据结构，包含多个字段时，可以使用这个输出解析器。该解析器可以生成指令帮助LLM返回结构化数据文本，同时完成文本到结构化数据的解析工作。示例代码如下：

```python
# 定义响应的结构(JSON)，两个字段 回答和 来源。
response_schemas = [
    ResponseSchema(name="回答", description="回答用户的问题"),
    ResponseSchema(name="来源", description="所提及的回答用户问题的来源，必须是一个网站。")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 获取响应格式化的指令
format_instructions = output_parser.get_format_instructions()
format_instructions
```

你应该期望能看到如下输出：
```shell
'The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":\n\n```json\n{\n\t"回答": string  // 回答用户的问题\n\t"来源": string  // 所提及的回答用户问题的来源，必须是一个网站。\n}\n```'
```

#### openai 聊天模型中使用

```python
chat_model = ChatOpenAI(temperature=0, model_name="qwen2.5:14b", openai_api_key='ollama', openai_api_base='http://localhost:11434/v1')

chat_prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("尽可能回答用户的问题。\n{format_instructions}\n{question}")  
    ],
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)

_input = chat_prompt.format_prompt(question="what's the capital of france?")
output = chat_model.invoke(_input.to_messages())
output_parser.parse(output.content)
```

你应该期望能看到如下输出：

```json
{'回答': 'The capital of France is Paris.',
 '来源': 'https://www.britannica.com/place/Paris'}
```

注，关于示例代码中引用的 `partial_variables`，请参考[Partial - Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/partial)。

## 总结
本节课程中，我们学习了什么是 `输出解析器` ，LangChain所支持的常见解析器，以及如何使用常见的两款解析器 `List Parser` 和 `Structured Output Parser`。随着LangChain的发展，应该会有更多解析器出现。

### 相关文档资料链接：
1. [Python Langchain官方文档](https://python.langchain.com/docs/introduction/) 
2. [Models - Langchain](https://python.langchain.com/docs/how_to/#chat-models)