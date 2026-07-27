*This project has been created as part of the 42 curriculum by aanouer.*

## Description
This project introduces function calling in Large Language Models (LLMs). The goal is to translate natural language prompts into structured, machine-executable function calls with typed arguments. By implementing **constrained decoding** on a small 0.6B parameter model (Qwen/Qwen3-0.6B), the project bridges the gap between natural language understanding and reliable, schema-compliant JSON generation, achieving 100% valid JSON output without relying on mere prompting.

## Instructions
**Installation:**
For 42 school sessions, first create the virtual environment by run `make setup` to install dependencies and caches saved on `goinfre`.

then activate it with the command bellow, change `login` with your login.
```bash
source /goinfre/login/callme/bin/activate
```

The project uses `uv` for dependency management. To install dependencies, use the provided Makefile:
```bash
make install
```

**Execution:**
Run the program from the root directory:
```bash
make run
```
Or directly using `uv`:
```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

## Resources
- **Function Calling & Constrained Decoding**: Techniques to restrict token generation to a specific grammar or schema.
- **LLM Tokenization**: Understanding how models like Qwen break down text using BPE/SentencePiece.
- **AI Usage**: AI tools were used to research constrained decoding algorithms, understand tokenization edge cases, and refine the logical flow of the token selection pipeline.

## Algorithm explanation
The implementation uses **constrained decoding** to ensure the LLM generates valid JSON matching the required schema. At each generation step, before token selection:
1. The model outputs logits for all possible next tokens.
2. The decoder identifies tokens that would maintain both valid JSON syntax and compliance with the `functions_definition.json` schema.
3. Invalid tokens' logits are set to negative infinity.
4. The next token is sampled exclusively from the remaining valid tokens.
This token-by-token guidance guarantees 100% syntactically and semantically correct JSON output.

## Design decisions
- **Pydantic**: Utilized for robust schema validation and strictly defining function definitions.
- **Token Masking**: Logits modification is handled dynamically at each step, mapping vocabulary tokens to their valid string representations.
- **Error Handling**: Graceful exception management ensures the program never crashes unexpectedly and provides clear user feedback on malformed inputs.

## Performance analysis
- **Accuracy**: Achieves 90%+ correct function selection and argument extraction.
- **Reliability**: Guarantees 100% parseable, schema-compliant JSON output, proving structural guidance overcomes raw model size limitations.
- **Speed**: Optimized to process all test prompts in under 5 minutes on standard hardware by efficiently computing token constraints.

## Challenges faced
- **Tokenization nuances**: Handling preceding spaces and special characters specific to the tokenizer. Solved by carefully mapping the vocabulary file.
- **Constraining nested structures**: Dynamically tracking the JSON parsing state during token generation. Solved by implementing logit masking that computes valid next tokens at each step based on the current JSON context (e.g., expecting a key vs. a value), setting all other token logits to negative infinity.
- **Constraining data types**: Ensuring the model outputs correctly formatted parameter values (strings, numbers, booleans). Solved by implementing dynamic logit masking that restricts token generation based on the expected parameter type.

## Testing strategy
- **Edge Cases**: Tested against empty strings, large numbers, special characters, and ambiguous prompts.
- **End-to-End Validation**: Processed `data/input/function_calling_tests.json` and ensured the output perfectly matched the expected schema in `data/output/function_calling_results.json`.
- **Schema Compliance**: Verified that function names and argument types in the output exactly match the definitions.

## Example usage
```bash
# Default usage reading from data/input/ and writing to data/output/
uv run python -m src

# Custom file paths
uv run python -m src \
  --functions_definition custom_functions.json \
  --input custom_prompts.json \
  --output custom_results.json
```
