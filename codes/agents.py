from codes.helpers import generate_response

class Agent:
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def execute(self, command):
        raise NotImplementedError

class CodeAgent(Agent):
    def execute(self, command):
        return generate_response(self.model, self.tokenizer, f"Write code to: {command}")

class APIAgent(Agent):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)

    def execute(self, command):
        return generate_response(self.model, self.tokenizer, f"Call API to: {command}")

class WebSearchAgent(Agent):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)

    def execute(self, command):
        return generate_response(self.model, self.tokenizer, f"Search web for: {command}")

def handle_task(task, model, tokenizer):
    if "code" in task.lower():
        agent = CodeAgent(model, tokenizer)
    elif "api" in task.lower():
        agent = APIAgent(model, tokenizer)
    elif "search" in task.lower():
        agent = WebSearchAgent(model, tokenizer)
    else:
        return "Unknown task type"
    
    return agent.execute(task)