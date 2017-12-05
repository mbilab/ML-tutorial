from itertools import permutations
from random import choice

def compare(a, b):
    A = 0
    B = {}
    for x, y in zip(a, b):
        if x == y:
            A += 1
        else:
            B[x] = 2 if x in B else 1
            B[y] = 2 if y in B else 1
    B = sum(1 for k in B if 2 == B[k])
    return "%sA%sB" % (A, B)

def delete_invalid_numbers(pool, guess, response):
    return [x for x in pool if compare(x, guess) == response]

def demo():
    pool = generate_numbers()
    while len(pool) > 1:
        guess = pick_a_number(pool)
        print("I guess " + ''.join(guess))
        response = input("Result: ")
        pool = delete_invalid_numbers(pool, guess, response)

    if 1 == len(pool):
        guess = pick_a_number(pool)
        print("Your number is " + ''.join(guess))
    else:
        print("You might input something wrong")

def generate_numbers():
    digits = [str(x) for x in range(10)]
    return [''.join(x) for x in permutations(digits, 4)]

def pick_a_number(pool):
    return choice(pool)

if '__main__' == __name__:
    demo()
