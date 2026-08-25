# Define the data
total_yes = 56
total_no = 44
prior_yes = total_yes / (total_yes + total_no)
prior_no = total_no / (total_yes + total_no)

# Feature counts for yes (1) in each class: [count_in_yes, count_in_no]
features = {
    "studied": [30, 10],
    "experience": [20, 5],
    "confident": [4, 1],
    "help": [13, 6],
    "retry": [26, 18]
}

# Questions in order
questions = [
    "I studied how to do it",
    "I have experience with this task",
    "I feel confident about succeeding",
    "Someone will help me",
    "I can retry if I fail"
]

# Map questions to feature keys
feature_keys = ["studied", "experience", "confident", "help", "retry"]

# Get user inputs
inputs = {}
for i, question in enumerate(questions):
    while True:
        answer = input(f"{question}? (yes/no): ").strip().lower()
        if answer in ['yes', 'no']:
            inputs[feature_keys[i]] = 1 if answer == 'yes' else 0
            break
        else:
            print("Please answer with 'yes' or 'no'.")

# Compute likelihoods
lik_yes = 1.0
lik_no = 1.0

for feature, value in inputs.items():
    count_yes, count_no = features[feature]
    
    if value == 1:
        p_yes = count_yes / total_yes
        p_no = count_no / total_no
    else:
        p_yes = (total_yes - count_yes) / total_yes
        p_no = (total_no - count_no) / total_no
    
    lik_yes *= p_yes
    lik_no *= p_no

# Compute posterior probability for yes
post_yes = (prior_yes * lik_yes) / (prior_yes * lik_yes + prior_no * lik_no)

# Output the probability
print(f"The probability of success is: {post_yes:.4f} or {post_yes * 100:.2f}%")

