"""
Functions used to randomly generate users according to a realistic distribution matching NYC

Name each function EXACTLY the same as the field name in `users.py`
"""

import random
def age():
    percentiles = [
        0.01, # up to age 1
        0.01, # up to age 2
        0.01, # up to age 3
        0.012, # ...
        0.012,
    ]
    r = random.random()
    cumulative = 0
    for i in range():
        cumulative += percentiles[i]
        if r < cumulative:
            return i