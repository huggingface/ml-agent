#!/usr/bin/env python3
"""Quick manual tester for Whoosh-backed search tools."""

import asyncio

from agent.tools.docs_tools import explore_hf_docs_handler, search_openapi_handler


async def test_docs_search() -> None:
    """Test HF docs search."""
    print("=" * 60)
    print("Testing explore_hf_docs (optimum, 'chat interface')")
    print("=" * 60)
    result, success = await explore_hf_docs_handler(
        arguments={
            "endpoint": "optimum",
            "query": "chat interface",
            "max_results": 5,
        }
    )
    print(f"Success: {success}")
    print(result[:1000] if len(result) > 1000 else result)


async def test_api_search() -> None:
    """Test OpenAPI search."""
    print("\n" + "=" * 60)
    print("Testing find_hf_api ('list user spaces')")
    print("=" * 60)
    result, success = await search_openapi_handler(
        arguments={
            "query": "list user spaces",
        }
    )
    print(f"Success: {success}")
    print(result[:2000] if len(result) > 2000 else result)


async def test_api_search_with_tag() -> None:
    """Test OpenAPI search with tag filter."""
    print("\n" + "=" * 60)
    print("Testing find_hf_api ('spaces', tag='spaces')")
    print("=" * 60)
    result, success = await search_openapi_handler(
        arguments={
            "query": "list",
            "tag": "spaces",
        }
    )
    print(f"Success: {success}")
    print(result[:2000] if len(result) > 2000 else result)


def main() -> None:
    # asyncio.run(test_docs_search())
    asyncio.run(test_api_search())
    # asyncio.run(test_api_search_with_tag())


if __name__ == "__main__":
    main()
