from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool, tool
from langchain.chat_models import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
import csv
import pandas as pd
import psycopg2

# Load environment variables from .env file
load_dotenv()

def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime  # Import datetime module to get current time

    now = datetime.datetime.now()  # Get current time
    return now.strftime("%I:%M %p")  # Format time in H:MM AM/PM format

def connect_to_database(*args, **kwargs):
    conn = psycopg2.connect(
        host="your_db_host",
        database="your_db_name",
        user="username",
        password="password",
        port='5432'
    )
    print("Connected to database successfully")

    return conn

def execute_query(query, *args, **kwargs):
    conn = connect_to_database()
    cursor = conn.cursor()
    cursor.execute(query)
    resp = cursor.fetchall()
    cursor.close()

    return resp

# List of tools available to the agent
tools = [
    Tool(
        name="Time",  # Name of the tool
        func=get_current_time,  # Function that the tool will execute
        # Description of the tool
        description="Useful for when you need to know the current time",
    ),
    Tool(
        name="database connect",  # Name of the tool
        func=connect_to_database,  # Function that the tool will execute
        # Description of the tool
        description="Useful for querying the SQL database to retrieve information"
    ),
    Tool(
        name="database query",  # Name of the tool
        func=execute_query,  # Function that the tool will execute
        # Description of the tool
        description="Useful for querying the SQL database to retrieve information. Only query from table public.result. The table contains id, regression_id, test_id, test_name, args, ran, result, duration, host_message, sut_message, error_line_count, timeout, setting, category, tst, jira_ticket, test_category, test_check, jira_status, usr_comment and updated.  Consindering the 'invalid' column. If it is invalid, do not take into account"
    )
]

class Agent:
    def __init__(self):
        # ReAct = Reason and Action
        # https://smith.langchain.com/hub/hwchase17/react
        template = """
        Answer the following questions as best you can. You have access to the following tools:
        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer. Make sure resolve this error psycopg2.errors.SyntaxError: unterminated quoted identifier at or near "" when dealing with psycopg2 database query. Should elimminate closing double quote at the end of the prompt
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
        """

        self.prompt = PromptTemplate.from_template(template)
        
        # Initialize a ChatOpenAI model
        llm = AzureChatOpenAI(
            deployment_name='swe-gpt4o-exp1', 
            temperature=0, 
            openai_api_version="2024-02-01", 
            openai_api_base="https://llm-api.amd.com", 
            openai_api_key="your_api_key", 
            model_kwargs = {
                'headers': { 
                    "Ocp-Apim-Subscription-Key": "f204857bfd0d46dc99335c36e406b066"
                }
                }
            )

        # Create the ReAct agent using the create_react_agent function
        self.agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=self.prompt,
            stop_sequence=True,
        )

        # Create an agent executor from the agent and tools
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=tools,
            verbose=True,
        )


if __name__ == "__main__":
    # Run the agent with a test query
    agentApi = Agent()
    
    response = agentApi.agent_executor.invoke({"input": "Can you summarise the result column for regression id 54097? "})

    # Print the response from the agent
    print("response:", response['output'])
