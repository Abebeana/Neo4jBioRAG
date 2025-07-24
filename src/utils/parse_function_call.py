from typing import List, Tuple

def parse_function_call(function_call: str) -> List[Tuple[str, str]]:
        """
        Parse function call string into list of (function_name, gene_name) tuples.

        Args:
            function_call: String containing function calls

        Returns:
            List of (function_name, gene_name) tuples

        Raises:
            ValueError: If function call format is invalid
        """
        try:
            # Split by newlines to handle multiple function calls
            calls = [call.strip() for call in function_call.split('\n') if call.strip()]
            parsed_calls = []

            for call in calls:
                if ':' in call:
                    function_name, gene_name = call.split(':')
                    parsed_calls.append((function_name.strip(), gene_name.strip()))

            if not parsed_calls:
                raise ValueError(f"No valid function calls found in: {function_call}")

            return parsed_calls
        except Exception as e:
            raise ValueError(f"Invalid function call format: {function_call}")

        