"""Registry patterns for medriskeval components.

This module provides generic registry classes for datasets, judges, models, and tasks.
Registries enable a plugin architecture where components can be registered and
retrieved by name, supporting extensibility and configuration-driven evaluation.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Type, TypeVar, overload
from abc import ABC, abstractmethod

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class RegistryError(Exception):
    """Exception raised for registry-related errors."""
    pass


class DuplicateRegistrationError(RegistryError):
    """Raised when attempting to register a name that already exists."""
    pass


class NotFoundError(RegistryError):
    """Raised when a requested name is not found in the registry."""
    pass


class BaseRegistry(Generic[T]):
    """Base registry class with common functionality.
    
    A registry maintains a mapping from string names to registered items
    (classes, functions, or instances). It provides methods to register,
    retrieve, and list items.
    
    Attributes:
        _registry: Internal mapping of names to registered items.
        _name: Human-readable name for this registry (used in error messages).
    """
    
    def __init__(self, name: str = "Registry") -> None:
        self._registry: dict[str, T] = {}
        self._name = name

    def register(self, name: str, item: T, *, allow_override: bool = False) -> T:
        """Register an item with the given name.
        
        Args:
            name: Unique identifier for this item.
            item: The item to register (class, function, or instance).
            allow_override: If True, allow overwriting existing registrations.
            
        Returns:
            The registered item (for decorator chaining).
            
        Raises:
            DuplicateRegistrationError: If name exists and allow_override is False.
        """
        if name in self._registry and not allow_override:
            raise DuplicateRegistrationError(
                f"{self._name}: '{name}' is already registered. "
                f"Use allow_override=True to replace."
            )
        self._registry[name] = item
        return item

    def get(self, name: str) -> T:
        """Retrieve a registered item by name.
        
        Args:
            name: The registered name to look up.
            
        Returns:
            The registered item.
            
        Raises:
            NotFoundError: If the name is not registered.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys())) or "(none)"
            raise NotFoundError(
                f"{self._name}: '{name}' not found. Available: {available}"
            )
        return self._registry[name]

    def contains(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._registry

    def list_names(self) -> list[str]:
        """Return a sorted list of all registered names."""
        return sorted(self._registry.keys())

    def items(self) -> list[tuple[str, T]]:
        """Return all registered name-item pairs."""
        return list(self._registry.items())

    def clear(self) -> None:
        """Remove all registrations (useful for testing)."""
        self._registry.clear()

    def __contains__(self, name: str) -> bool:
        return self.contains(name)

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"{self._name}({self.list_names()})"


class ClassRegistry(BaseRegistry[Type[T]]):
    """Registry for classes with decorator support.
    
    Provides a decorator pattern for easy class registration:
    
        @registry.register("my_name")
        class MyClass:
            ...
    """
    
    @overload
    def register(self, name: str) -> Callable[[Type[T]], Type[T]]: ...
    
    @overload
    def register(self, name: str, cls: Type[T], *, allow_override: bool = False) -> Type[T]: ...
    
    def register(
        self, 
        name: str, 
        cls: Type[T] | None = None, 
        *, 
        allow_override: bool = False
    ) -> Type[T] | Callable[[Type[T]], Type[T]]:
        """Register a class, either directly or as a decorator.
        
        Can be used as:
            registry.register("name", MyClass)
        Or as a decorator:
            @registry.register("name")
            class MyClass: ...
        """
        if cls is not None:
            return super().register(name, cls, allow_override=allow_override)
        
        def decorator(cls: Type[T]) -> Type[T]:
            super(ClassRegistry, self).register(name, cls, allow_override=allow_override)
            return cls
        return decorator

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        """Create an instance of a registered class.
        
        Args:
            name: The registered class name.
            *args: Positional arguments for the constructor.
            **kwargs: Keyword arguments for the constructor.
            
        Returns:
            A new instance of the registered class.
        """
        cls = self.get(name)
        return cls(*args, **kwargs)


class FunctionRegistry(BaseRegistry[Callable[..., T]]):
    """Registry for functions/factories with decorator support.
    
    Provides a decorator pattern for easy function registration:
    
        @registry.register("my_func")
        def my_function(arg):
            ...
    """
    
    @overload
    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]: ...
    
    @overload
    def register(
        self, name: str, func: Callable[..., T], *, allow_override: bool = False
    ) -> Callable[..., T]: ...
    
    def register(
        self,
        name: str,
        func: Callable[..., T] | None = None,
        *,
        allow_override: bool = False
    ) -> Callable[..., T] | Callable[[Callable[..., T]], Callable[..., T]]:
        """Register a function, either directly or as a decorator."""
        if func is not None:
            return super().register(name, func, allow_override=allow_override)
        
        def decorator(fn: Callable[..., T]) -> Callable[..., T]:
            super(FunctionRegistry, self).register(name, fn, allow_override=allow_override)
            return fn
        return decorator

    def call(self, name: str, *args: Any, **kwargs: Any) -> T:
        """Call a registered function by name.
        
        Args:
            name: The registered function name.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
            
        Returns:
            The return value of the function.
        """
        func = self.get(name)
        return func(*args, **kwargs)


# =============================================================================
# Global registries for medriskeval components
# =============================================================================

# Dataset registry: maps benchmark names to dataset loader classes
DatasetRegistry: ClassRegistry[Any] = ClassRegistry("DatasetRegistry")

# Judge registry: maps judge names to judge classes
JudgeRegistry: ClassRegistry[Any] = ClassRegistry("JudgeRegistry")

# Model registry: maps model names to model adapter classes  
ModelRegistry: ClassRegistry[Any] = ClassRegistry("ModelRegistry")

# Task registry: maps task names to factory functions that create task pipelines
TaskRegistry: FunctionRegistry[Any] = FunctionRegistry("TaskRegistry")

# Metric registry: maps metric names to metric computation functions
MetricRegistry: FunctionRegistry[Any] = FunctionRegistry("MetricRegistry")

# Prompt registry: maps prompt names to prompt template classes/functions
PromptRegistry: ClassRegistry[Any] = ClassRegistry("PromptRegistry")


def list_all_registries() -> dict[str, list[str]]:
    """Return a summary of all registered components."""
    return {
        "datasets": DatasetRegistry.list_names(),
        "judges": JudgeRegistry.list_names(),
        "models": ModelRegistry.list_names(),
        "tasks": TaskRegistry.list_names(),
        "metrics": MetricRegistry.list_names(),
        "prompts": PromptRegistry.list_names(),
    }
