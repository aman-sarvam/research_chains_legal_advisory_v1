import httpx
import asyncio

# async def query_ask_sarvam(body):
#     url = "https://law-dev.sarvam.ai/api/backend/regchat/query"
#     async with httpx.AsyncClient() as client:
#         response = await client.post(url, json=body)
#         response.raise_for_status()
#         return response.json()

async def query_ask_sarvam(body):
    url = "https://law-dev.sarvam.ai/api/backend/regchat/query"
    # Set a longer timeout period, e.g., 30 seconds
    timeout = httpx.Timeout(120.0, connect=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=body)
        response.raise_for_status()
        return response.json()

async def main():
    thread = {
        "thread" : [
            {
                "role": "user",
                "content": "What are the current regulations regarding the import and use of SIM cards in vehicles in India? This is all the information I have. Give me a summary of everything",
            },
        ]
    }
    resp = await query_ask_sarvam(thread)
    print(resp)

if __name__ == "__main__":
    asyncio.run(main())
