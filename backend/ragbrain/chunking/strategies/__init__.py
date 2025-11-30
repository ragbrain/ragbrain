"""Chunking strategies - Auto-register all strategies"""

from ragbrain.chunking.factory import ChunkingStrategyFactory
from ragbrain.chunking.recursive import RecursiveChunkingStrategy
from ragbrain.chunking.markdown import MarkdownChunkingStrategy
from ragbrain.chunking.character import CharacterChunkingStrategy
from ragbrain.chunking.semantic import SemanticChunkingStrategy
from ragbrain.chunking.transcript import TranscriptChunkingStrategy
from ragbrain.chunking.hierarchical import HierarchicalChunkingStrategy

# Register all strategies
ChunkingStrategyFactory.register('recursive', RecursiveChunkingStrategy)
ChunkingStrategyFactory.register('markdown', MarkdownChunkingStrategy)
ChunkingStrategyFactory.register('character', CharacterChunkingStrategy)
ChunkingStrategyFactory.register('semantic', SemanticChunkingStrategy)
ChunkingStrategyFactory.register('transcript', TranscriptChunkingStrategy)
ChunkingStrategyFactory.register('hierarchical', HierarchicalChunkingStrategy)

__all__ = [
    'RecursiveChunkingStrategy',
    'MarkdownChunkingStrategy',
    'CharacterChunkingStrategy',
    'SemanticChunkingStrategy',
    'TranscriptChunkingStrategy',
    'HierarchicalChunkingStrategy',
]
