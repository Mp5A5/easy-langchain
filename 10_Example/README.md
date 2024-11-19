---
title: 10. 一个完整的例子
tags:
  - openai
  - llm
  - langchain
---

# Langchain极简入门: 10. 一个完整的例子

-----

## 简介

这是该 `LangChain` 极简入门系列的最后一讲。我们将利用过去9讲学习的知识，来完成一个具备完整功能集的LLM应用。该应用基于 `LangChain` 框架，以某 `PDF` 文件的内容为知识库，提供给用户基于该文件内容的问答能力。

我们利用 `LangChain` 的QA chain，结合 `Chroma` 来实现PDF文档的语义化搜索。示例代码所引用的是[AWS Serverless
Developer Guide](https://docs.aws.amazon.com/pdfs/serverless/latest/devguide/serverless-core.pdf)，该PDF文档共84页。

本讲的完整代码请参考[10_Example.jpynb](./10_Example.ipynb)

1. 安装必要的 `Python` 包

    ```shell
    pip install langchain-ollama==0.2.0
    pip install chromadb 
    pip install pymupdf 
    pip install tiktoken
    ```

3. 下载PDF文件AWS Serverless Developer Guide

    ```python
    !wget https://docs.aws.amazon.com/pdfs/serverless/latest/devguide/serverless-core.pdf

    PDF_NAME = 'serverless-core.pdf'
    ```

3. 加载PDF文件

    ```python
    from langchain_community.document_loaders import PyMuPDFLoader
    
    pdf_path = './serverless-core.pdf'
    docs = PyMuPDFLoader(pdf_path).load()
    
    print (f'There are {len(docs)} document(s) in {pdf_path}.')
    print (f'There are {len(docs[0].page_content)} characters in the first page of your document.')
    ```

4. 拆分文档并存储文本嵌入的向量数据

    ```python
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma.vectorstores import Chroma
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="/shibing624/text2vec-base-chinese",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    vectorstore = Chroma.from_documents(split_docs, embeddings, collection_name="serverless_guide")
    ```

5. 基于OpenAI创建QA链

    ```python
    from langchain_core.prompts import (
        ChatPromptTemplate,
        PromptTemplate,
        SystemMessagePromptTemplate,
        AIMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
    
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Context: {context} 
    Question: {question} 
    """
    
    retriever = vectorstore.as_retriever()
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    ## 利用 ChatPromptTemplate.from_messages
    # human_message_prompt = HumanMessagePromptTemplate(
    #     prompt=PromptTemplate(
    #         template=template,
    #         input_variables=["question", "context"],
    #     )
    # )
    
    # prompt = ChatPromptTemplate.from_messages([human_message_prompt])
    
    ## 利用 ChatPromptTemplate.from_template
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(temperature=0, model_name="qwen2.5:14b", openai_api_key='ollama', openai_api_base='http://localhost:11434/v1')
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    ```

8. 基于相关文档，利用QA链完成回答

    ```python
    rag_chain.invoke("What is the use case of AWS Serverless?")
    ```

## 总结

本节课程中，我们利用所学的知识，完成了第一个完整的LLM应用。希望通过本系列的学习，大家能对 `LangChain` 框架的使用，有了基本的认识，并且掌握了框架核心组建的使用方法。

### 相关文档资料链接：
1. [Python Langchain官方文档](https://python.langchain.com/docs/introduction/) 
2. [Models - Langchain](https://python.langchain.com/docs/how_to/#chat-models)