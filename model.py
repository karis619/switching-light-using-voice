class student:
    def __init__(self, name, course):
        
     self.name = name
     self.course = course


    def introduce(self):
        return f"I am {self.name}"

s = student("Amina","AI")
print(s.introduce())

