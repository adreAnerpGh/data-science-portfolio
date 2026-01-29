# Calling Ollama to synthesise a dataset of documents (50), at least 500 words in length 
# Types of documents: news articles.
# Saving each article to an individual local file. 
# Ensuring that the content created does contain Named Entities.

# Setup Ollama connection
from langchain_ollama import OllamaLLM
# Import the operating system module to interact with files and directories
import os

# Initialize the model
model = OllamaLLM(model="gemma:2b")

# List of topics 
# https://en.wikipedia.org/wiki/Portal:Current_events/October_2025
topics = [
    "Royal succession in Luxembourg",
    "First female Archbishop of Canterbury",
    "AI policy and regulation in the United States",
    "Climate change and hurricanes in the Caribbean",
    "Syrian parliamentary elections",
    "New space exploration missions to Mars",
    "Innovations in sustainable transportation in Asia",
    "Cybersecurity incidents in multinational companies",
    "Healthcare technology advancements",
    "Global economic trends and market recovery",
    "The impact of the war in Ukraine on global energy markets",
    "The consequences of climate change for biodiversity",      
    "The latest developments in the global AI landscape",
    "The rise of China and its implications for the US economy",
    "The ongoing humanitarian crisis in Yemen",
    "The ongoing political and economic standoff between Israel and the Palestinians",
    "The future of democracy in the face of rising populism",
    "The global response to the COVID-19 pandemic",
    "The role of technology in combating climate change",
    "The ethical considerations surrounding artificial intelligence",
    "The challenges of global food security",
    "The ongoing struggle for global peace and security",
    "The impact of the war in Ukraine on the global economy",
    "The future of the European Union",
    "The ongoing conflict in the Middle East",
    "The latest developments in the global space race",
    "The human rights violations committed by governments around the world",
    "The growing movement for sustainable development",
    "The challenges and opportunities of the digital world",
    "The impact of the war in Ukraine on the global arts and culture",
    "The future of the world of work",
    "The impact of technology on social justice",
    "The challenges of global health",
    "The global refugee crisis",
    "The impact of climate change on the natural environment",
    "The challenges of global education",
    "The future of the global economy",
    "The challenges of global poverty",
    "The impact of technology on democracy",
    "The role of women in the global economy",
    "The challenges of global governance",
    "The impact of technology on culture",
    "The challenges of global communication",
    "The future of the global health care system",
    "The impact of technology on society",
    "The challenges of global education",
    "The future of the global economy",
    "The challenges of global migration",
    "The impact of technology on the environment",
    "The challenges of global poverty"
]

# Ensure the folder exists
os.makedirs("synthetic_docs", exist_ok=True)

# Generate and save documents
# https://realpython.com/python-enumerate
for i, topic in enumerate(topics, start=1):  
    # enumerate gives both the index (i) and the topic string
    # start=1 ensures numbering begins at 1 instead of 0

    #  Prompt for news articles
    prompt = f"""
You are an expert journalist. Write one original news article of approximately 500–600 words.
Include multiple Named Entities: people, locations, organisations and dates.
Topic: {topic}.
Write in natural prose, no lists or headings.
"""       
    content = model.invoke(prompt)

    # Save file in the manually created folder
    filename = f"synthetic_docs/doc_{i:02d}.txt"  # formats the number as 2  digits with leading zero if necessary
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content.strip())

    print(f"Saved: {filename}")