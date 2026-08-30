class ConversationMemory:

    def __init__(self):
        self.messages = []

    def add(self, role, content):
        self.messages.append({
            "role": role,
            "content": content
        })

    def get_messages(self):
        return self.messages

    def clear(self):
        self.messages = []