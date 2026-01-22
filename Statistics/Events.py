

import random

# Simulate 10,000 die rolls
n_trials = 10_000
rolls = [random.randint(1, 6) for _ in range(n_trials)]

# Count occurrences
count_even = sum(1 for r in rolls if r in event_even)
count_gt_3 = sum(1 for r in rolls if r in event_gt_3)
count_prime = sum(1 for r in rolls if r in event_prime)

# Empirical probabilities
P_even_empirical = count_even / n_trials
P_gt_3_empirical = count_gt_3 / n_trials
P_prime_empirical = count_prime / n_trials

print("\nEmpirical Probabilities (from simulation):")
print(f"P(even)   ≈ {P_even_empirical:.4f}")
print(f"P(> 3)    ≈ {P_gt_3_empirical:.4f}")
print(f"P(prime)  ≈ {P_prime_empirical:.4f}")
