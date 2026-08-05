from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = "John" # default values
    age: Optional[int] = None
    grade: str
    email: EmailStr
    cgpa: float = Field(ge=0.0, le=4.0)

new_student = Student(age = '14', grade="A", email="abc@gmail.com", cgpa=3.5)
print(new_student)