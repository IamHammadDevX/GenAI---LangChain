from langchain_core.prompts import ChatPromptTemplate, MessagePlaceholder, load_prompt

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent.'),
    ('human', '{query}')
])