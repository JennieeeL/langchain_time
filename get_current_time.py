from datetime import datetime
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.agents import Tool

# Define the custom function to get the current time
def get_current_time():
    """Returns the current time as a string in HH:MM:SS format."""
    return datetime.now().strftime("%H:%M:%S")

# Define the custom function to get the current date
def get_current_date():
    """Returns the current date as a string in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")

# Create tools for the time and date functions
current_time_tool = Tool(
    name="CurrentTime",
    func=get_current_time,
    description="Use this tool to get the current time."
)

current_date_tool = Tool(
    name="CurrentDate",
    func=get_current_date,
    description="Use this tool to get the current date."
)

# Set up the LangChain LLM (you can replace this with your LLM model, such as OpenAI's GPT)
llm = OpenAI(openai_api_key="you_api_key", temperature=0.7)

# Set up the agent with the time and date tools
tools = [current_time_tool, current_date_tool]
agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Function to check if the user's question is about the current time or date
def is_time_or_date_related_question(user_input):
    """Check if the user's question is related to time or date."""
    time_keywords = ["time", "hour", "clock", "current time", "what time is it"]
    date_keywords = ["date", "today", "current date", "what's the date", "what is today's date"]
    
    if any(keyword.lower() in user_input.lower() for keyword in time_keywords):
        return "time"
    elif any(keyword.lower() in user_input.lower() for keyword in date_keywords):
        return "date"
    else:
        return None

# Define the function to interact with the agent
def get_agent_response(user_input):
    """Uses the LangChain agent to get a response based on the user's input."""
    # Check if the question is related to time or date
    category = is_time_or_date_related_question(user_input)
    
    if category == "time":
        # Call the time tool to get the current time
        return current_time_tool.func()
    elif category == "date":
        # Call the date tool to get the current date
        return current_date_tool.func()
    else:
        # If not time or date-related, pass the input to the agent for general response
        return agent.run(user_input)

# Example of usage:
if __name__ == "__main__":
    while True:
        # Ask the user for input (this could be a question or any text)
        user_input = input("Ask me anything (type 'exit' to stop): ")
        
        if user_input.lower() == 'exit':
            break
        
        # Get the agent's response
        response = get_agent_response(user_input)
        print(f"Response: {response}")
