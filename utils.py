# type: ignore
import os
from typing import TextIO

import openai
import pandas as pd
import streamlit as st
from langchain_experimental.agents import AgentExecutor, create_csv_agent, create_pandas_dataframe_agent
from langchain.llms import OpenAI

#openai.api_key = st.secrets["OPENAI_API_KEY"]


def get_answer_csv(file: TextIO, query: str) -> str:
    """
    Returns the answer to the given query by querying a CSV file.

    Args:
    - file (str): the file path to the CSV file to query.
    - query (str): the question to ask the agent.

    Returns:
    - answer (str): the answer to the query from the CSV file.
    """
    # Load the CSV file as a Pandas dataframe
    # df = pd.read_csv(file)
    #df = pd.read_csv("titanic.csv")

    # Create an agent using OpenAI and the Pandas dataframe
    agent = create_csv_agent(OpenAI(temperature=0), file, verbose=False, allow_dangerous_code=True, model='gpt-4-turbo')
    #agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=False)
    
    # Wrap the agent with the AgentExecutor to handle parsing errors
    executor = AgentExecutor(agent=agent, handle_parsing_errors=True)


    # Run the agent on the given query and return the answer
    #query = "whats the square root of the average age?"
    answer = agent.run(query)
    return answer
