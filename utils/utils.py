def get_validated_input(message: str, options: dict, defaults: dict = None):
    user_inputs = {}

    print(message)

    for option, expected_type in options.items():
        while True:
            default_value = defaults.get(option) if defaults and option in defaults else None
            default_text = f" (default is {default_value})" if default_value is not None else ""

            # Prompt user, showing default if available
            user_input = input(f"{option}{default_text} (type {expected_type.__name__}): ").strip()

            # Use the default value if user presses Enter
            if user_input == "" and default_value is not None:
                user_inputs[option] = default_value
                break

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

    return user_inputs
