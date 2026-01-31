"""
Run large-scale validation test with 100 matches
"""
import asyncio
from src.evaluation.batch_evaluator import run_batch_evaluation

async def main():
    await run_batch_evaluation(num_matches=100, use_ensemble=False)

if __name__ == "__main__":
    asyncio.run(main())
