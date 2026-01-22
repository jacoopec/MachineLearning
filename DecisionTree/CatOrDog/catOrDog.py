#This is an handcrafted (manual) rule-based decision tree Based on domain knowledge,
# not a machine-learned one.
#So structurally, it is a decision tree — just not one that’s learned from data.

def classify_animal(
    has_whiskers: bool, #Baffi
    ear_shape: str,           # "pointed" or "floppy"
    snout_length: str,        # "short", "medium", "long"  #muso
    eye_shape: str,           # "vertical" or "round"
    size_relative_to_face: str  # "small", "medium", "large"
) -> str:
    """
    Classifies an animal as a cat or dog based on facial features.
    """

    if not has_whiskers:
        return "dog"

    if ear_shape == "pointed":
        if eye_shape == "vertical":
            return "cat"
        elif snout_length == "long":
            return "dog"
        else:
            return "cat"
    else:  # floppy ears
        if snout_length == "long" or size_relative_to_face == "large":
            return "dog"
        else:
            return "cat"

# Example 1: Cat
animal = classify_animal(
    has_whiskers=True,
    ear_shape="pointed",
    snout_length="short",
    eye_shape="vertical",
    size_relative_to_face="small"
)
print(animal)  # Output: cat

# Example 2: Dog
animal = classify_animal(
    has_whiskers=True,
    ear_shape="floppy",
    snout_length="long",
    eye_shape="round",
    size_relative_to_face="large"
)
print(animal)  # Output: dog
