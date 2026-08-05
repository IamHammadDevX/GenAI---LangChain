from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    city: str

person1: Person = {'name': 'Alice', 'age': 30, 'city': 'New York'}

print(person1)