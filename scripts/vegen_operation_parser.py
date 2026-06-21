#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class BitSlice:
    bv: Any
    hi: Any
    lo: Any


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Number:
    val: int | float


@dataclass(frozen=True)
class Update:
    lhs: Any
    rhs: Any


@dataclass(frozen=True)
class For:
    iterator: str
    begin: Any
    end: Any
    body: list[Any]
    inc: bool


@dataclass(frozen=True)
class If:
    cond: Any
    then: list[Any]
    otherwise: list[Any]


@dataclass(frozen=True)
class Call:
    func: str
    args: list[Any]


@dataclass(frozen=True)
class BinaryExpr:
    op: str
    a: Any
    b: Any
    expr_id: int


@dataclass(frozen=True)
class UnaryExpr:
    op: str
    a: Any


@dataclass(frozen=True)
class PseudoStmt:
    desc: str


@dataclass(frozen=True)
class Lookup:
    obj: Any
    key: str


@dataclass
class ParserState:
    expr_id: int = 0

    def next_expr_id(self) -> int:
        self.expr_id += 1
        return self.expr_id


TOKEN_RE = re.compile(
    r"0x[0-9a-fA-F]+|\d+\.\d+|\d+|:=|<-|←|==|!=|<=|>=|<<<|>>>|<<|>>|&&|\|\||"
    r"[A-Za-z_][A-Za-z0-9_]*|[-+*/%<>&|^~!?:.,()[\]{}=]"
)

BINARY_PRECEDENCE: dict[str, tuple[int, str]] = {
    "||": (1, "OR"),
    "OR": (1, "OR"),
    "|": (2, "|"),
    "XOR": (3, "XOR"),
    "^": (3, "XOR"),
    "&&": (4, "AND"),
    "AND": (4, "AND"),
    "&": (5, "&"),
    "==": (6, "=="),
    "!=": (6, "!="),
    "<": (7, "<"),
    ">": (7, ">"),
    "<=": (7, "<="),
    ">=": (7, ">="),
    "<<": (8, "<<"),
    ">>": (8, ">>"),
    "<<<": (8, "<<<"),
    ">>>": (8, ">>>"),
    "+": (9, "+"),
    "-": (9, "-"),
    "*": (10, "*"),
    "/": (10, "/"),
    "%": (10, "%"),
}

ASSIGN_TOKENS = {":=", "<-", "←", "="}
END_TOKENS = {"ENDFOR", "FI", "ENDIF", "ELSE"}


def normalize_source(text: str) -> str:
    text = text.replace("鈫恷=", ":=").replace("鈫?", ":=").replace("鈫怓", ":=")
    text = text.replace("鈫怲", ":=").replace("鈫怋", ":=").replace("鈫恓", ":=")
    text = text.replace("←", ":=")
    text = re.sub(r"//.*", "", text)
    text = re.sub(r";", "\n", text)
    return text


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text) if token.strip()]


class ExpressionParser:
    def __init__(self, tokens: list[str], state: ParserState) -> None:
        self.tokens = tokens
        self.state = state
        self.pos = 0

    def peek(self) -> str | None:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise SyntaxError("unexpected end of expression")
        self.pos += 1
        return token

    def consume(self, token: str) -> bool:
        if self.peek() == token:
            self.pos += 1
            return True
        return False

    def parse(self) -> Any:
        if not self.tokens:
            return Var("empty")
        expr = self.parse_expr(0)
        return expr

    def parse_expr(self, min_prec: int) -> Any:
        left = self.parse_prefix()
        while True:
            token = self.peek()
            lookup = token.upper() if token and token.isalpha() else token
            if lookup not in BINARY_PRECEDENCE:
                break
            prec, op = BINARY_PRECEDENCE[lookup]
            if prec < min_prec:
                break
            self.take()
            right = self.parse_expr(prec + 1)
            left = BinaryExpr(op, left, right, self.state.next_expr_id())
        return left

    def parse_prefix(self) -> Any:
        token = self.take()
        upper = token.upper()
        if token == "(":
            expr = self.parse_expr(0)
            if not self.consume(")"):
                raise SyntaxError("missing ')'")
            return self.parse_postfix(expr)
        if token in {"-", "!", "~"}:
            return UnaryExpr(token, self.parse_expr(11))
        if re.fullmatch(r"0x[0-9a-fA-F]+", token):
            return self.parse_postfix(Number(int(token, 16)))
        if re.fullmatch(r"\d+\.\d+", token):
            return self.parse_postfix(Number(float(token)))
        if re.fullmatch(r"\d+", token):
            return self.parse_postfix(Number(int(token)))
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            if self.consume("("):
                args = self.parse_args(")")
                return self.parse_postfix(Call(token, args))
            return self.parse_postfix(Var(token))
        raise SyntaxError(f"unexpected token in expression: {token}")

    def parse_postfix(self, expr: Any) -> Any:
        while True:
            if self.consume("."):
                expr = Lookup(expr, self.take())
                continue
            if self.consume("["):
                hi = self.parse_expr(0)
                lo = hi
                if self.consume(":"):
                    lo = self.parse_expr(0)
                if not self.consume("]"):
                    raise SyntaxError("missing ']'")
                expr = BitSlice(expr, hi=hi, lo=lo)
                continue
            break
        return expr

    def parse_args(self, end: str) -> list[Any]:
        args: list[Any] = []
        if self.consume(end):
            return args
        while True:
            args.append(self.parse_expr(0))
            if self.consume(end):
                return args
            if not self.consume(","):
                raise SyntaxError(f"expected ',' or '{end}'")


def parse_expr_text(text: str, state: ParserState | None = None) -> Any:
    parser = ExpressionParser(tokenize(text), state or ParserState())
    return parser.parse()


def parse_assignment(line: str, state: ParserState) -> Update | None:
    tokens = tokenize(line)
    depth = 0
    for index, token in enumerate(tokens):
        if token in {"(", "["}:
            depth += 1
        elif token in {")",
            "]",
        }:
            depth -= 1
        elif depth == 0 and token in ASSIGN_TOKENS:
            lhs = ExpressionParser(tokens[:index], state).parse()
            rhs = ExpressionParser(tokens[index + 1 :], state).parse()
            return Update(lhs, rhs)
    return None


def split_operation_lines(operation: str | list[str]) -> list[str]:
    if isinstance(operation, list):
        text = "\n".join(str(line) for line in operation)
    else:
        text = str(operation)
    normalized = normalize_source(text)
    lines: list[str] = []
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if line:
            lines.append(line)
    return lines


def parse_block(lines: list[str], pos: int, state: ParserState, stop: set[str]) -> tuple[list[Any], int, str | None]:
    stmts: list[Any] = []
    while pos < len(lines):
        line = lines[pos].strip()
        head = line.split(None, 1)[0].upper() if line else ""
        if head == "ENDIF":
            head = "FI"
        if head in stop:
            return stmts, pos, head
        if head == "FOR":
            stmt, pos = parse_for(lines, pos, state)
            stmts.append(stmt)
            continue
        if head == "IF":
            stmt, pos = parse_if(lines, pos, state)
            stmts.append(stmt)
            continue
        try:
            assignment = parse_assignment(line, state)
        except SyntaxError:
            assignment = None
        stmts.append(assignment if assignment is not None else PseudoStmt(line))
        pos += 1
    return stmts, pos, None


def parse_for(lines: list[str], pos: int, state: ParserState) -> tuple[For, int]:
    line = lines[pos].strip()
    match = re.match(r"FOR\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?::=|=|<-|←)\s*(.+?)\s+(TO|DOWNTO)\s+(.+)$", line, re.IGNORECASE)
    if match is not None:
        iterator, begin_text, direction, end_text = match.groups()
    else:
        range_match = re.match(
            r"FOR\s+([A-Za-z_][A-Za-z0-9_]*)\s+IN\s+([-+]?\d+)\s*\.\.\s*([-+]?\d+)$",
            line,
            re.IGNORECASE,
        )
        if range_match is not None:
            iterator, begin_text, end_text = range_match.groups()
            direction = "TO"
        else:
            domain_match = re.match(
                r"FOR\s+([A-Za-z_][A-Za-z0-9_]*)\s+IN\s+(.+)$",
                line,
                re.IGNORECASE,
            )
            if domain_match is None:
                raise SyntaxError(f"unsupported FOR header: {line}")
            iterator, domain_text = domain_match.groups()
            begin_text = "0"
            end_text = domain_text
            direction = "TO"
    body, next_pos, stop = parse_block(lines, pos + 1, state, {"ENDFOR"})
    if stop != "ENDFOR":
        raise SyntaxError(f"FOR without ENDFOR: {line}")
    return For(
        iterator=iterator,
        begin=parse_expr_text(begin_text, state),
        end=parse_expr_text(end_text, state),
        body=body,
        inc=direction.upper() == "TO",
    ), next_pos + 1


def parse_if(lines: list[str], pos: int, state: ParserState) -> tuple[If, int]:
    line = lines[pos].strip()
    cond_text = re.sub(r"^IF\s+", "", line, flags=re.IGNORECASE)
    cond_text = re.sub(r"\s+THEN$", "", cond_text, flags=re.IGNORECASE).strip()
    cond_text = re.sub(r"\s+is\s+true$", "", cond_text, flags=re.IGNORECASE).strip()
    false_match = re.search(r"\s+is\s+false$", cond_text, flags=re.IGNORECASE)
    if false_match:
        cond_text = "!(" + cond_text[: false_match.start()].strip() + ")"
    then_body, next_pos, stop = parse_block(lines, pos + 1, state, {"ELSE", "FI"})
    otherwise: list[Any] = []
    if stop == "ELSE":
        otherwise, next_pos, stop = parse_block(lines, next_pos + 1, state, {"FI"})
    if stop != "FI":
        raise SyntaxError(f"IF without FI: {line}")
    return If(parse_expr_text(cond_text, state), then_body, otherwise), next_pos + 1


def parse_operation_ast(operation: str | list[str]) -> list[Any]:
    state = ParserState()
    lines = split_operation_lines(operation)
    stmts, pos, stop = parse_block(lines, 0, state, set())
    if pos != len(lines) or stop is not None:
        raise SyntaxError("unexpected parser stop")
    return stmts


def ast_to_json(value: Any) -> Any:
    if isinstance(value, (str, int, float)) or value is None:
        return value
    if isinstance(value, list):
        return [ast_to_json(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {
            "kind": type(value).__name__,
            **{name: ast_to_json(getattr(value, name)) for name in value.__dataclass_fields__},
        }
    return str(value)
