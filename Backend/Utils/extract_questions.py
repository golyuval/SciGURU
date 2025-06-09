import json
import ijson  # for streaming JSON parsing

def extract_questions():
    questions = []
    with open('Backend/Utils/evaluated_preference_data.json', 'r', encoding='utf-8') as f:
        # Use ijson to parse the file in chunks
        parser = ijson.parse(f)
        current_prompt = ""
        for prefix, event, value in parser:
            if prefix == 'item.prompt':
                current_prompt = value
                start = current_prompt.find('user<|end_header_id|>\n') + len('user<|end_header_id|>\n')
                end = current_prompt.find('<|eot_id|>', start)
                question = current_prompt[start:end].strip()
                questions.append(question)
    
    with open('Backend/Utils/questions.py', 'w', encoding='utf-8') as f:
        f.write("questions = [\n")
        for question in questions:
            f.write(f'    "{question}",\n')
        f.write("]")

if __name__ == "__main__":
    extract_questions() 