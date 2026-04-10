from dotenv import load_dotenv
import re 
import inspect

load_dotenv()

import ollama
from langsmith import traceable


MAX_INTERATIONS = 10
MODEL = ""

# --- Tools (LangChain @tool decorator) ---

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"   >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 199.99, "keyboard": 89.95}
    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"   >> Executing apply_discount(price='{price}, discount_tier='{discount_tier}')")
    price = float(price)
    discount_percentages = {"bronze": 9, "silver": 15, "gold": 25}
    discount = discount_percentages.get(discount_tier,  0)
    return round(price * (1 - discount/100), 2)

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

# CHANGE 3: Delete the JSON schemas. Tools now live inside the prompt as plain text.
# We derive description from the functions themselves using inspect.
def get_tool_discriptions(tools_dict): 
    discriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g., @traceable add *, config=None)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoct(original_function) or ""
        discriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(discriptions)

tool_descriptions = get_tool_discriptions(tools) # injected to LLM prompt
tool_names = ", ".join(tools.keys)

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.
# CHANGE 4: Drop tools= from ollama.chat(). The LLM has no idea it's an agent -
# all agency come from the prompt above and our regex parsing below.

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)

# --- Agent Loop ---

@traceable(name="LangChain Agent Loop")
def run_agent(question: str) :
    print(f"Quetions: {question}")
    print("="*60)
    
    # CHANGE 5: One prompt string replaces the system/user message split
    prompt = react_prompt.format(question=question)
    scratchpad = "" # for LLM history
    for interartion in range(1, MAX_INTERATIONS * 1):
        print("\n---Interation {iteration} ---")
        full_prompt = prompt + scratchpad
        #ai_message = llm_with_tools.invoke(messages)
        
        # Difference5: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(
            odel=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        
        output = response.message.content
        print(f"LLM Ouput: \n{output}")
        
        print(f"    [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search("Final Answer: \s*(.+)", output)
        if final_answer_match :
            final_answer = final_answer_match.group(1).strip()
            print(f"    [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" *60)
            print(f"Final Answer: {final_answer}")
        
        # CHANGE 6: Parse tool calls from raw text with reges - fragile if LLM does not follow format.
        print(f"    [Parsing] Looking for Action and Action Input in LLM output...")
        
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break
        
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")
        
        # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]
        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))

        print(f"  [Tool Result] {observation}")

        # CHANGE 7: History is one growing string re-sent every iteration (replaces messages.append).
        scratchpad += f"{output}\nObservation: {observation}\nThought:"


    print("ERROR: Max iterations reached without a final answer")
    return None
        
if __name__ == "__main__" :
    print("Hello LangChain Agent {.bind_tools}")
    result = run_agent("What is the price of a laptop after applying a gold disacount?")
    
    