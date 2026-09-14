"""
Backward Compatibility Proxy for Query Router.
Re-exports QueryRouter, RouteCategory, RouteDecision from src.core.router.
"""

from src.core.router import QueryRouter, RouteCategory, RouteDecision

__all__ = ["QueryRouter", "RouteCategory", "RouteDecision"]
