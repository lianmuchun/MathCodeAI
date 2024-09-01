import streamlit as st  # 导入 Streamlit 库，用于创建 Web 应用界面
import logging  # 导入 logging 库，用于记录日志
import os  # 导入 os 库，用于操作文件和路径
import tempfile  # 导入 tempfile 库，用于创建临时文件和目录
import shutil  # 导入 shutil 库，用于文件操作，如复制和删除
import pdfplumber  # 导入 pdfplumber 库，用于处理 PDF 文件
import ollama  # 导入 ollama 库，用于 Ollama 模型的接口

# 导入 LangChain 相关模块，用于处理文档加载、嵌入生成、文本分割、向量存储等功能
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional  # 导入 typing 模块，用于类型注释

# Streamlit 页面配置
st.set_page_config(
    page_title="MathCode AI",  # 页面标题
    page_icon="🚀",  # 页面图标
    layout="wide",  # 页面布局为宽屏
    initial_sidebar_state="collapsed",  # 侧边栏默认状态为收起
)

with st.sidebar:
    st.markdown("# 💡 关于")
    st.divider()
    st.markdown("**架构⚙️**：*Ollama + Streamlit + LangChain*")
    st.markdown("**团队🥇**：数码宝贝")
    st.markdown("**作者👦🏻**：李智琛 & 刘红斌 & 卢兴湛 & 曹俊泽 & 周济坤")
    st.markdown("**作者👩🏻**：梁思杰 & 周洁")
    st.markdown("**代码仓库💻**：*https://gitee.com/liang-sijie/code-babies*")

# 日志配置
logging.basicConfig(
    level=logging.INFO,  # 日志级别为 INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
    datefmt="%Y-%m-%d %H:%M:%S",  # 日期格式
)

logger = logging.getLogger(__name__)  # 获取日志记录器

# 缓存函数，用于提取模型名称
@st.cache_resource(show_spinner=True)
def extract_model_names(
    models_info: Dict[str, List[Dict[str, Any]]],  # 输入参数：模型信息字典
) -> Tuple[str, ...]:

    logger.info("Extracting model names from models_info")  # 记录提取模型名称的日志
    model_names = tuple(model["name"] for model in models_info["models"])  # 提取模型名称
    logger.info(f"Extracted model names: {model_names}")  # 记录提取的模型名称
    return model_names  # 返回模型名称元组

# 创建向量数据库的函数
def create_vector_db(file_upload) -> Chroma:

    logger.info(f"Creating vector DB from file upload: {file_upload.name}")  # 记录创建向量数据库的日志
    temp_dir = tempfile.mkdtemp()  # 创建临时目录

    path = os.path.join(temp_dir, file_upload.name)  # 定义临时文件路径
    with open(path, "wb") as f:  # 打开临时文件进行写操作
        f.write(file_upload.getvalue())  # 将上传的文件内容写入临时文件
        logger.info(f"File saved to temporary path: {path}")  # 记录文件保存路径
        loader = UnstructuredPDFLoader(path)  # 创建 PDF 加载器
        data = loader.load()  # 加载 PDF 内容

    # 创建文本分割器，将文档分割成块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)  # 将文档分割成块
    logger.info("Document split into chunks")  # 记录文档分割完成的日志

    # 创建 Ollama 嵌入生成器
    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5", show_progress=True)
    vector_db = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name="myRAG"  # 创建向量数据库
    )
    logger.info("Vector DB created")  # 记录向量数据库创建完成的日志

    shutil.rmtree(temp_dir)  # 删除临时目录
    logger.info(f"Temporary directory {temp_dir} removed")  # 记录临时目录删除的日志
    return vector_db  # 返回向量数据库对象

# 处理用户问题的函数
def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:

    logger.info(f"Processing question: {question} using model: {selected_model}")  # 记录处理问题的日志
    llm = ChatOllama(model=selected_model, temperature=0)  # 创建聊天模型实例
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],  # 输入变量
        template="""You are an AI language model assistant. Your task is to generate 3
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",  # 提问提示模板
    )

    # 使用多查询检索器
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Only provide the answer from the {context}, nothing else.
    Add snippets of the context you used to answer the question.
    """

    prompt = ChatPromptTemplate.from_template(template)  # 创建聊天提示模板

    # 创建处理链，将上下文、问题、模型和输出解析器连接起来
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)  # 执行处理链，生成回答
    logger.info("Question processed and response generated")  # 记录问题处理完成的日志
    return response  # 返回生成的回答

# 缓存函数，用于提取 PDF 文件的所有页面作为图像
@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:

    logger.info(f"Extracting all pages as images from file: {file_upload.name}")  # 记录提取 PDF 页面的日志
    pdf_pages = []  # 初始化页面列表
    with pdfplumber.open(file_upload) as pdf:  # 打开 PDF 文件
        pdf_pages = [page.to_image().original for page in pdf.pages]  # 将每个页面提取为图像
    logger.info("PDF pages extracted as images")  # 记录页面提取完成的日志
    return pdf_pages  # 返回页面图像列表

# 删除向量数据库的函数
def delete_vector_db(vector_db: Optional[Chroma]) -> None:

    logger.info("Deleting vector DB")  # 记录删除向量数据库的日志
    if vector_db is not None:
        vector_db.delete_collection()  # 删除向量数据库集合
        st.session_state.pop("pdf_pages", None)  # 清除会话状态中的 PDF 页面数据
        st.session_state.pop("file_upload", None)  # 清除会话状态中的文件上传数据
        st.session_state.pop("vector_db", None)  # 清除会话状态中的向量数据库
        st.success("Collection and temporary files deleted successfully.")  # 显示删除成功消息
        logger.info("Vector DB and related session state cleared")  # 记录向量数据库删除完成的日志
        st.rerun()  # 重新运行应用程序
    else:
        st.error("No vector database found to delete.")  # 显示错误消息，提示未找到向量数据库
        logger.warning("Attempted to delete vector DB, but none was found")  # 记录未找到向量数据库的警告日志

# 主函数，用于运行 Streamlit 应用程序
def main() -> None:

    st.title("🚀 MathCode AI")
    st.subheader("", divider="gray", anchor=False)  # 设置应用标题

    models_info = ollama.list()  # 获取可用模型的信息
    available_models = extract_model_names(models_info)  # 提取模型名称

    col1, col2 = st.columns([1.5, 2])  # 创建两个列布局

    if "messages" not in st.session_state:  # 如果会话状态中没有消息，则初始化
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:  # 如果会话状态中没有向量数据库，则初始化
        st.session_state["vector_db"] = None

    if available_models:  # 如果有可用模型
        selected_model = col2.selectbox(
            "🌈 **选择模型**", available_models  # 在列 2 中创建模型选择框
        )

    file_upload = col1.file_uploader(
        "📁 **上传 PDF 文件**", type="pdf", accept_multiple_files=False  # 在列 1 中创建文件上传器
    )

    if file_upload:  # 如果文件已上传
        st.session_state["file_upload"] = file_upload  # 保存上传的文件到会话状态
        if st.session_state["vector_db"] is None:  # 如果向量数据库未创建
            st.session_state["vector_db"] = create_vector_db(file_upload)  # 创建向量数据库
        pdf_pages = extract_all_pages_as_images(file_upload)  # 提取 PDF 页面为图像
        st.session_state["pdf_pages"] = pdf_pages  # 保存页面图像到会话状态

        zoom_level = col1.slider(
            "Zoom Level", min_value=100, max_value=1000, value=700, step=50  # 在列 1 中创建缩放级别滑块
        )

        with col1:  # 在列 1 中显示 PDF 页面图像
            with st.container(height=410, border=True):
                for page_image in pdf_pages:  # 遍历每个页面图像
                    st.image(page_image, width=zoom_level)  # 显示图像，使用缩放级别

    delete_collection = col1.button("⚠️ **删除会话**", type="secondary")  # 在列 1 中创建删除集合按钮

    if delete_collection:  # 如果按下删除按钮
        delete_vector_db(st.session_state["vector_db"])  # 删除向量数据库

    with col2:  # 在列 2 中显示消息
        message_container = st.container(height=500, border=True)

        for message in st.session_state["messages"]:  # 遍历会话状态中的消息
            avatar = "🤖" if message["role"] == "assistant" else "😎"  # 根据消息角色设置头像
            with message_container.chat_message(message["role"], avatar=avatar):  # 在消息容器中显示消息
                st.markdown(message["content"])  # 显示消息内容

        if prompt := st.chat_input("请输入您的问题..."):  # 如果用户输入了提示
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})  # 将用户消息添加到会话状态
                message_container.chat_message("user", avatar="😎").markdown(prompt)  # 显示用户消息

                with message_container.chat_message("assistant", avatar="🤖"):  # 显示助手消息
                    with st.spinner(":green[processing...]"):  # 显示处理中的提示
                        if st.session_state["vector_db"] is not None:  # 如果向量数据库已创建
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model  # 处理用户问题
                            )
                            st.markdown(response)  # 显示助手回答
                        else:
                            st.warning("请先上传 PDF 文件")  # 提示用户先上传 PDF 文件

                if st.session_state["vector_db"] is not None:  # 如果向量数据库已创建
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}  # 将助手消息添加到会话状态
                    )

            except Exception as e:  # 捕获异常
                st.error(e, icon="⛔️")  # 显示错误消息
                logger.error(f"Error processing prompt: {e}")  # 记录错误日志
        else:
            if st.session_state["vector_db"] is None:  # 如果未上传 PDF 文件
                st.warning("开始聊天前请先上传 PDF 文件...")  # 提示用户上传 PDF 文件

# 程序入口，运行主函数
if __name__ == "__main__":
    main()