from dotenv import load_dotenv

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
    discount_percentages = {"bronze": 9, "silver": 15, "gold": 25}
    discount = discount_percentages.get(discount_tier,  0)
    return round(price * (1 - discount/100), 2)

# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)  
# --- Agent Loop ---

@traceable(name="LangChain Agent Loop")
def run_agent(question: str) :
    tools = [get_product_price, apply_discount]
    #tools_dict = {t.name: t for t in tools}
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }
    #llm = init_chat_model(f"ollama:{MODEL}", temparature = 0)    
    #llm_with_tools = llm.bind_tools(tools)
    
    print(f"Quetions: {question}")
    print("="*60)
    
    # Only work for Ollama
    messages = [
        {
            "role": "system",
            "content": {
                "You are a helpful shopping assistant."
                "You have access to a product catalog too "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            }
        },
        {
            "role": "user",
            "content": question
        }
    ]
    
    for interartion in range(1, MAX_INTERATIONS * 1):
        print("\n---Interation {iteration} ---")
        
        #ai_message = llm_with_tools.invoke(messages)
        
        # Difference5: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls
        # If no tool calls, this is the final anwer
        if not tool_calls :
            print("\nFinal Answer: {ai_message.content}")
            return ai_message.content
        # Process only the first tool call - force one tool per iteration. In fact, multiple tools can be processed. 
        tool_call = tool_calls[0]
        #tool_name = tool_call.get("name")
        #tool_args = tool_call.get("args", {})
        #tool_id = tool_call.get("id")  # for tracibility
        
        # Difference 6: Attribute access {.function.name} instead of {.get("name")}
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        
        print(f"   [Tool Selected] {tool_name} with args: {tool_args}")
        
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        # Difference 7: Direct function call instead of tool.invoke()
        #observation = tool_to_use.invoke(tool_args)
        observation = tool_to_use(**tool_args)
        
        print(f" | [Tool Result] {observation}")
        # Add the history
        messages.append(ai_message)
        messages.append(
            {
            #ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                "role": "tool",
                "content": str(observation)
            }
        )
    print("ERROR")
    
if __name__ == "__main__" :
    print("Hello LangChain Agent {.bind_tools}")
    result = run_agent("What is the price of a laptop after applying a gold disacount?")
    
    