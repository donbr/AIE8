# %% [markdown]
# # LangGraph and LangSmith - Agentic RAG Powered by LangChain

# %% [markdown]
# ## Common Setup

# %%
import os
import getpass
from uuid import uuid4

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

os.environ["TAVILY_API_KEY"] = getpass.getpass("TAVILY_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AIE8 - LangGraph - {uuid4().hex[0:8]}"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API Key: ")

# %%
from langchain_openai import ChatOpenAI

default_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
helpfulness_llm = ChatOpenAI(model="gpt-4.1-mini")

# %%
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langchain_community.tools.arxiv.tool import ArxivQueryRun

tavily_tool = TavilySearch(max_results=5)

tool_belt = [
    tavily_tool,
    ArxivQueryRun(),
]

model_with_tools = default_llm.bind_tools(tool_belt)

# %%
from langgraph.prebuilt import ToolNode

def call_model(state):
  messages = state["messages"]
  response = model_with_tools.invoke(messages)
  return {"messages" : [response]}

tool_node = ToolNode(tool_belt)

# %% [markdown]
# ## Simple Agentic Graph

# %%
from langgraph.graph import StateGraph, END

def should_continue(state):
  last_message = state["messages"][-1]

  if last_message.tool_calls:
    return "action"

  return END

# %%
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
  messages: Annotated[list, add_messages]

# %%
simple_graph = StateGraph(AgentState)

simple_graph.add_node("agent", call_model)
simple_graph.add_node("action", tool_node)

simple_graph.add_conditional_edges(
    "agent",
    should_continue
)

simple_graph.add_edge("action", "agent")

simple_graph.set_entry_point("agent")

# %%
simple_graph_app = simple_graph.compile()

# %%
inputs = {"messages" : [HumanMessage(content="How are technical professionals using AI to improve their work?")]}

async for chunk in simple_graph_app.astream(inputs, stream_mode="updates"):
    for node, values in chunk.items():
        print(f"Receiving update from node: '{node}'")
        print(values["messages"])
        print("\n\n")

# %%
inputs = {"messages" : [HumanMessage(content="Search Arxiv for the A Comprehensive Survey of Deep Research paper, then search each of the authors to find out where they work now using Tavily!")]}

async for chunk in simple_graph_app.astream(inputs, stream_mode="updates"):
    for node, values in chunk.items():
        print(f"Receiving update from node: '{node}'")
        if node == "action":
          print(f"Tool Used: {values['messages'][0].name}")
        print(values["messages"])
        print("\n\n")

# %% [markdown]
# ## Simple Agentic Graph with LangSmith Evaluation

# %%
def convert_inputs(input_object):
  return {"messages" : [HumanMessage(content=input_object["text"])]}

def parse_output(input_state):
  return {"answer" : input_state["messages"][-1].content}

agent_chain_with_formatting = convert_inputs | simple_graph_app | parse_output

agent_chain_with_formatting.invoke({"text" : "What is Deep Research?"})

# %%
questions = [
    {
        "inputs" : {"text" : "Who were the main authors on the 'A Comprehensive Survey of Deep Research: Systems, Methodologies, and Applications' paper?"},
        "outputs" : {"must_mention" : ["Peng", "Xu"]}   
    },
    {
        "inputs" : {"text" : "Where do the authors of the 'A Comprehensive Survey of Deep Research: Systems, Methodologies, and Applications' work now?"},
        "outputs" : {"must_mention" : ["Zhejiang", "Liberty Mutual"]}
    }
]

# %%
from langsmith import Client

client = Client()

dataset_name = f"Simple Search Agent - Evaluation Dataset - {uuid4().hex[0:8]}"

dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Questions about the cohort use-case to evaluate the Simple Search Agent."
)

client.create_examples(
    dataset_id=dataset.id,
    examples=questions
)

# %%
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
# print(CORRECTNESS_PROMPT)

correctness_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="openai:o3-mini", # very impactful to the final score
        feedback_key="correctness",
    )

# %%
def must_mention(inputs: dict, outputs: dict, reference_outputs: dict) -> float:
  # determine if the phrases in the reference_outputs are in the outputs
  required = reference_outputs.get("must_mention") or []
  score = all(phrase in outputs["answer"] for phrase in required)
  return score

# %%
results = client.evaluate(
    agent_chain_with_formatting,
    data=dataset.name,
    evaluators=[correctness_evaluator, must_mention],
    experiment_prefix="simple_agent, baseline",  # optional, experiment name prefix
    description="Testing the baseline system.",  # optional, experiment description
    max_concurrency=4, # optional, add concurrency
)

# %% [markdown]
# ## Agentic Graph with LLM Helpfulness Check

# %%
HELPFULNESS_PROMPT_TMPL = """\
Given an initial query and a final response, determine if the final response is extremely helpful or not. Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.

Initial Query:
{initial_query}

Final Response:
{final_response}"""

# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

def tool_call_or_helpful(state):
  last_message = state["messages"][-1]

  if last_message.tool_calls:
    return "action"

  initial_query = state["messages"][0]
  final_response = state["messages"][-1]

  if len(state["messages"]) > 10:
    return "END"

  helpfullness_prompt_template = PromptTemplate.from_template(HELPFULNESS_PROMPT_TMPL)

  helpfulness_chain = helpfullness_prompt_template | helpfulness_llm | StrOutputParser()

  helpfulness_response = helpfulness_chain.invoke({"initial_query" : initial_query.content, "final_response" : final_response.content})

  if "Y" in helpfulness_response:
    return "end"
  else:
    return "continue"

# %%
class AgentState(TypedDict):
  messages: Annotated[list, add_messages]

# %%
helpfulness_graph = StateGraph(AgentState)

helpfulness_graph.add_node("agent", call_model)
helpfulness_graph.add_node("action", tool_node)

helpfulness_graph.add_edge("action", "agent")

helpfulness_graph.add_conditional_edges(
    "agent",
    tool_call_or_helpful,
    {
        "continue" : "agent",
        "action" : "action",
        "end" : END
    }
)

helpfulness_graph.set_entry_point("agent")

# %%
helpfulness_graph_app = helpfulness_graph.compile()

# %%
inputs = {"messages" : [HumanMessage(content="What are Deep Research Agents?")]}

async for chunk in helpfulness_graph_app.astream(inputs, stream_mode="updates"):
    for node, values in chunk.items():
        print(f"Receiving update from node: '{node}'")
        print(values["messages"])
        print("\n\n")

# %%
patterns = ["Context Engineering", "Fine-tuning", "LLM-based agents"]

for pattern in patterns:
  what_is_string = f"What is {pattern} and when did it break onto the scene??"
  inputs = {"messages" : [HumanMessage(content=what_is_string)]}
  messages = helpfulness_graph_app.invoke(inputs)
  print(messages["messages"][-1].content)
  print("\n\n")


