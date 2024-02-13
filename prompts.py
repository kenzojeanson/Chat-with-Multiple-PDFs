from openai import OpenAI
import env
API_KEY = env.API_KEY
client = OpenAI(api_key=API_KEY)  # API Key

# ---------- Get Answer ---------- #
def get_answer(question, text_chunk):
    template = f'''   
    [Start of Document]: \n
    "{text_chunk}" \n
    [End of Document] \n
    
    [Start of Question]: \n
    {question} \n
    [End of Question]: \n \n    
    
    [Instructions]: \n
    Step 1: The question was given above between [Start of Question] and [End of Question]. Pick one of the following Templates depending on the [Question], Do not add this as part of your response. But follow the template: \n
        1. If the question is relevant to the text documents, use [Template #1] \n
        2. If the question is not relevant to the text documents, use [Template #2] \n
        3. If the question is not in the form of a question, or you do not know how to reply, use [Template #3] \n \n
                       
    [Template #1]: \n
        First; Repeat back the user question. Do not include quotation marks such as "" or '' \n
        Next; Answer the question confidently, accurately, and with detail \n
        
        Example: \n    
        User Question: "What is the mass of the sun?" \n
        Structure your output like this: \n
        
        Question: What is the mass of the sun? \n  
        Answer: The mass of the sun is 3.955 × 10^30 kg. \n \n

    [Template #2]: \n
        First; Repeat back the user question \n
        Next; Answer with: "Please ensure that your message is a question relevant to the information provided in the document." \n
        
        Example: \n    
        User Question: "Irrelevant question" \n
        Structure your output like this: \n
        
        Question: "Irrelevant Question" \n  
        Answer: "Please ensure that your message is a question relevant to the information provided in the document." \n \n

    [Template #3]: \n
        First; Repeat back the user question \n
        Next; Answer with: "Please ensure your message in the form of a question relevant to the information provided in the document." \n
        
        Example: \n    
        User Question: "Not a question" \n
        Structure your output like this: \n
        
        Question: "Not a Question" \n  
        Answer: "Please ensure your message in the form of a question relevant to the information provided in the document." \n \n
        
        
    Step 2: In your response, ensure the following: \n
        1. The name of the template is not in the response. \n
        2. Only use what is specifically provided in the templates. Only use one template at a time. \n \n
        
       
    [Special Instruction] \n
    Let's structure our response. Let's think step by step.    
    '''

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": template},
        ],
        temperature=0,
        max_tokens=1000,
    )

    answer = response.choices[0].message.content

    assert response.usage is not None
    prompt_cost = response.usage.prompt_tokens * 0.01 / 1000
    completion_cost = response.usage.completion_tokens * 0.03 / 1000
    total_cost = round(prompt_cost + completion_cost, 3)

    print("Total Cost: $", total_cost)

    return answer
