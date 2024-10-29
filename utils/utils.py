def get_validated_input(message: str, options: dict):

    user_inputs = {}

    print(f"{message} (or type 'exit' to quit)")

    for option, expected_type in options.items():
        while True:
            user_input = input(f"Please enter {option} (type {expected_type.__name__}): ").strip()

            if user_input.lower() == 'exit':
                print("Exiting the program.")
                return None

            # Validate and type cast the input
            if expected_type == int:
                if user_input.isdigit() or (user_input.startswith('-') and user_input[1:].isdigit()):
                    user_inputs[option] = int(user_input)
                    break
                else:
                    print(f"Invalid input. '{option}' should be an integer. Please try again.")

            elif expected_type == float:
                try:
                    user_inputs[option] = float(user_input)
                    break
                except ValueError:
                    print(f"Invalid input. '{option}' should be a float. Please try again.")

            elif expected_type == str:
                if user_input:  # Ensure the string is not empty
                    user_inputs[option] = user_input
                    break
                else:
                    print(f"Invalid input. '{option}' should be a non-empty string. Please try again.")

            else:
                print(f"Unsupported type '{expected_type.__name__}' for option '{option}'.")
                return None

    return user_inputs
