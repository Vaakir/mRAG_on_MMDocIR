"""Tool definitions for System 3 agents (optional - tools are currently embedded in nodes)."""

# In this implementation, "tools" are the System 2 components:
# - QueryTechnique classes for query rewriting/retrieval
# - PromptStrategy classes for answer generation
#
# These are accessed directly from:
# - query_techniques package (System 2)
# - generation.prompts package (System 2)
#
# If you want to expose these as proper LangChain tools,
# you can implement that here. For now, agents use direct function calling.
