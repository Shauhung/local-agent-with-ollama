import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

# 1. 直接使用 OllamaModel
# 確保你已經執行過 ollama run qwen2.5-coder
model = OllamaModel(
    model_name='qwen2.5-coder:32b',
    provider=OllamaProvider(base_url='http://localhost:11434/v1')
)

# 2. 定義一個簡單的工具 (Tool)
# 注意：工具必須加上型別提示 (Type Hints) 和 Docstring，Agent 才會知道怎麼用
def get_system_time() -> str:
    """回傳目前的系統時間"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 3. 初始化 Agent
agent = Agent(
    model,
    system_prompt="你是一位專業的助理，如果有需要，可以使用提供的工具。",
    tools=[get_system_time]
)

async def main():
    history = []
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['exit', 'quit']: break
            
        try:
            # 1. 執行並取得結果
            result = await agent.run(user_input, message_history=history)
            
            # 2. 如果 result 裡面有工具執行結果，它會自動將其加入下一次的訊息中
            # 確保我們使用的是 result.data (這是最終內容)
            print(f"Agent: {result.output}")
            
            # 3. 更新歷史紀錄
            history = list(result.all_messages())
            
        except Exception as e:
            # 這裡幫你把錯誤印出來，看看是不是 JSON 解析錯誤
            print(f"發生錯誤: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())