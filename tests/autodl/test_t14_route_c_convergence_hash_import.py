"""Exercise the exact delayed expression without launching a Route-C owner."""

import ast
import hashlib
import inspect
import json

from scripts.autodl import run_t14_route_c_owner as owner


def test_deployed_convergence_receipt_expression_has_all_imports():
    source = ast.parse(inspect.getsource(owner))
    expression = next(
        node.value for node in ast.walk(source)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "convergence"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "audit_sha256"
            for target in node.targets
        )
    )
    fixture = {"status": "NOT_CONVERGED", "audited_at": "fixed", "owner_pid": 1}
    # Execute only the actual hash expression. No resource query or main()
    # call, so a missing late-stage module import is caught on a tiny fixture.
    actual = eval(compile(ast.Expression(expression), "owner_hash", "eval"),
                  vars(owner), {"convergence": fixture})
    expected = hashlib.sha256(json.dumps(fixture, sort_keys=True,
                                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert actual == expected
