from mypy.plugin import Plugin, ClassDefContext
from mypy.nodes import FuncDef, Decorator, OverloadedFuncDef, FakeInfo
from mypy.parse import parse
from mypy.options import Options
from typing import List, Type

def symtrace_class_maker_callback(ctx: ClassDefContext) -> None:
    """
    @traceable handler

    Given an `nn.Module` definition with @traceable applied, transform the
    AST of that module during type checking to create an overload of the
    `forward` method that takes `Proxy` objects instead of the normal types.
    This allows for discovery of untraceable constructs during the type
    checking phase rather than at runtime.


    Example. Given this Module:

    @traceable
    class Foo(torch.nn.Module):
        def forward(self, x : torch.Tensor) -> torch.Tensor:
            return torch.neg(x)

    Yield this module:

    @traceable
    class Foo(torch.nn.Module):
        @overload
        def forward(self, x : torch.fx.proxy.Proxy) -> torch.fx.proxy.Proxy: ...

        @overload
        def forward(self, x : torch.Tensor) -> torch.Tensor: ...

        def forward(self, x : Union[torch.Tensor, torch.fx.proxy.Proxy]) -> torch.fx.proxy.Proxy:
            return torch.neg(x)
    """
    stmts = ctx.cls.defs.body
    forward_idx = -1
    for i, stmt in enumerate(stmts):
        if isinstance(stmt, OverloadedFuncDef):
            return
        if isinstance(stmt, FuncDef) and stmt.name == 'forward':
            forward_idx = i
            break

    if forward_idx == -1:
        raise RuntimeError('@traceable class must have forward() defined!')

    forward_fn = stmts[forward_idx]

    # TODO: enforce that all parameters and returns (except self) are type
    # annotated
    # TODO: return types
    arg_exprs_real = []
    arg_exprs_proxy = []
    for i, arg in enumerate(forward_fn.arguments):
        name = arg.variable.name
        arg_exprs_real.append(f'{name}')
        type_annotation = ' : torch.fx.proxy.Proxy' if i > 0 else ''
        arg_exprs_proxy.append(f'{name}{type_annotation}')

    # Create Decorator node for proxy overload
    fn_def_proxy = f"""
@overload
def forward({', '.join(arg_exprs_proxy)}): ...
"""

    parsed_proxy = parse(fn_def_proxy, '<string>', None, None, Options())
    proxy_decorator = None
    for def_ in parsed_proxy.defs:
        if isinstance(def_, Decorator):
            proxy_decorator = def_
            break

    # Create dceorator node for normal overload
    fn_def_real = f"""
@overload
def forward({', '.join(arg_exprs_real)}): ...
"""
    parsed_real = parse(fn_def_real, '<string>', None, None, Options())
    real_decorator = None
    for def_ in parsed_real.defs:
        if isinstance(def_, Decorator):
            real_decorator = def_
            break

    real_args = [arg.type_annotation for arg in real_decorator.func.arguments]
    proxy_args = [arg.type_annotation for arg in proxy_decorator.func.arguments]

    for k in {'type', 'unanalyzed_type'}:
        setattr(real_decorator.func, k, getattr(forward_fn, k, None))

    real_decorator.func.arguments = forward_fn.arguments

    # TODO: fixup `forward` implementation to take and return the union of
    # Proxy and the originally-annotated types

    overloaded = OverloadedFuncDef([proxy_decorator, real_decorator, forward_fn])
    stmts[forward_idx] = overloaded

    # Defer to get typechecking run on the newly-generated AST
    ctx.api.defer()

class SymTracePlugin(Plugin):
    def get_class_decorator_hook(self, fullname: str):
        if fullname == 'test.traceable':
            return symtrace_class_maker_callback
        return None

def plugin(version: str):
    return SymTracePlugin
