import asyncio

from agent.tools.dataset_tools import hf_inspect_dataset_handler, inspect_dataset
from agent.tools.docs_tools import (
    explore_hf_docs_handler,
    hf_docs_fetch_handler,
    search_openapi_handler,
)
from agent.tools.github_find_examples import find_examples, github_find_examples_handler
from agent.tools.github_list_repos import github_list_repos_handler, list_repos
from agent.tools.github_read_file import github_read_file_handler, read_file
from agent.tools.hf_repo_files_tool import hf_repo_files_handler
from agent.tools.hf_repo_git_tool import hf_repo_git_handler
from agent.tools.jobs_tool import hf_jobs_handler
from agent.tools.plan_tool import get_current_plan, plan_tool_handler
from agent.tools.private_hf_repo_tools import private_hf_repo_handler


# Dataset tools
async def test_inspect_dataset():
    result = await inspect_dataset(dataset="HuggingFaceFW/finetranslations")
    print(result["formatted"], len(result["formatted"]))


async def test_hf_inspect_dataset_handler():
    result, success = await hf_inspect_dataset_handler(
        {"dataset": "HuggingFaceFW/finetranslations"}
    )
    print(result, success)


# GitHub tools
def test_list_repos():
    result = list_repos(owner="huggingface", owner_type="org", sort="stars", limit=5)
    print(result["formatted"], len(result["formatted"]))


async def test_github_list_repos_handler():
    result, success = await github_list_repos_handler(
        {"owner": "huggingface", "owner_type": "org", "sort": "stars", "limit": 5}
    )
    print(result, success)


def test_read_file():
    result = read_file(
        repo="huggingface/transformers",
        path="/src/transformers/loss/loss_for_object_detection.py",
    )
    print(result["formatted"], len(result["formatted"]))


async def test_github_read_file_handler():
    result, success = await github_read_file_handler(
        {"repo": "huggingface/transformers", "path": "README.md"}
    )
    print(result, success)


def test_find_examples():
    result = find_examples(
        keyword="sft",
        repo="transformers",
        org="huggingface",
        max_results=5,
        min_score=40,
    )
    print(result["formatted"], len(result["formatted"]))


async def test_github_find_examples_handler():
    result, success = await github_find_examples_handler(
        {"keyword": "grpo", "repo": "trl", "org": "huggingface", "max_results": 5}
    )
    print(result, success)


async def test_explore_hf_docs_handler():
    result, success = await explore_hf_docs_handler({"endpoint": "trl"})
    print(result, success)


async def test_search_openapi_handler():
    result, success = await search_openapi_handler({"tag": "spaces", "query": "logs"})
    print(result, success)


async def test_hf_docs_fetch_handler():
    result, success = await hf_docs_fetch_handler(
        {"url": "https://huggingface.co/docs/trl/main/en/sft_trainer"}
    )
    print(result, success)


# Jobs tool
async def test_hf_jobs_handler():
    result, success = await hf_jobs_handler({"operation": "ps"})
    print(result, success)


# Plan tool
async def test_plan_tool_handler():
    result, success = await plan_tool_handler(
        {"todos": [{"id": "1", "content": "Test task", "status": "pending"}]}
    )
    print(result, success)


def test_get_current_plan():
    plan = get_current_plan()
    print(plan)


# Private HF Repo tools
async def test_private_hf_repo_handler():
    result, success = await private_hf_repo_handler(
        {"operation": "list", "repo_id": "test-repo", "repo_type": "dataset"}
    )
    print(result, success)


# HF Repo Files tool
async def test_hf_repo_files_handler():
    result, success = await hf_repo_files_handler(
        {"operation": "list", "repo_id": "bert-base-uncased", "repo_type": "model"}
    )
    print(result, success)


# HF Repo Git tool
async def test_hf_repo_git_handler():
    result, success = await hf_repo_git_handler(
        {"operation": "status", "repo_id": "test-repo", "repo_type": "model"}
    )
    print(result, success)


if __name__ == "__main__":
    # Uncomment the test you want to run:
    # asyncio.run(test_inspect_dataset())
    # test_list_repos()
    # asyncio.run(test_github_list_repos_handler())
    # test_read_file()
    # asyncio.run(test_github_read_file_handler())
    # test_search_code()
    # asyncio.run(test_github_search_code_handler())
    # test_find_examples()
    # asyncio.run(test_github_find_examples_handler())
    # asyncio.run(test_explore_hf_docs()) # definitely issues
    # asyncio.run(test_explore_hf_docs_handler())
    asyncio.run(test_search_openapi_handler())
    # asyncio.run(test_hf_docs_fetch_handler())
    # asyncio.run(test_hf_jobs_handler())
    # asyncio.run(test_plan_tool_handler())
    # test_get_current_plan()
    # asyncio.run(test_private_hf_repo_handler())
    # asyncio.run(test_hf_repo_files_handler())
    # asyncio.run(test_hf_repo_git_handler())
