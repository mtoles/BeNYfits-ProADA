import random

def sample_from_distribution(dist):
    total_weight = sum(weight for (_, weight) in dist)
    r = random.uniform(0, total_weight)

    cumulative = 0
    for (low_high, weight) in dist:
        low, high = low_high 
        if cumulative + weight >= r:
            print(f"low: {low}, high: {high}")
            if isinstance(low, int) and isinstance(high, int):
                return random.randint(low, high)
            else:
                return random.uniform(low, high)            
        cumulative += weight
    
    last_range, _ = dist[-1]
    low, high = last_range
    if isinstance(low, int) and isinstance(high, int):
        return random.randint(low, high)
    else:
        return random.uniform(low, high)

def sample_categorical(dist):
    total_weight = sum(weight for (_, weight) in dist)
    r = random.uniform(0, total_weight)

    cumulative = 0
    for (label, weight) in dist:
        if cumulative + weight >= r:
            return label
        cumulative += weight

    return dist[-1][0]


# Example usage
dist = [((0, 4), 5.4), ((5, 9), 5.4), ((10, 14), 5.6), ((15, 19), 5.7), ((20, 24), 7.0), ((25, 29), 9.0), ((30, 34), 8.8), ((35, 39), 7.4), ((40, 44), 6.5), ((45, 49), 6.0), ((50, 54), 6.2), ((55, 59), 6.2), ((60, 64), 5.8), ((65, 69), 4.8), ((70, 74), 3.9), ((75, 79), 2.6), ((80, 84), 1.8), ((85, 100), 1.9)]
print(sample_from_distribution(dist))

dist = [((0.0, 4.0), 5.4),
 ((5.0, 9.0), 5.4),
 ((10.0, 14.0), 5.6),
 ((15.0, 19.0), 5.7),
 ((20.0, 24.0), 7.0),
 ((25.0, 29.0), 9.0),
 ((30.0, 34.0), 8.8),
 ((35.0, 39.0), 7.4),
 ((40.0, 44.0), 6.5),
 ((45.0, 49.0), 6.0),
 ((50.0, 54.0), 6.2),
 ((55.0, 59.0), 6.2),
 ((60.0, 64.0), 5.8),
 ((65.0, 69.0), 4.8),
 ((70.0, 74.0), 3.9),
 ((75.0, 79.0), 2.6),
 ((80.0, 84.0), 1.8),
 ((85.0, 100.0), 1.9)]

print(sample_from_distribution(dist))

dist = [("Yes", 1), ("No", 99)]
print(sample_categorical(dist))