from typing import List, Any
from enum import Enum, auto

class TokenType(Enum):
    # Type
    NUMBER = auto()

    # Operator
    PLUS = auto()
    MINUS = auto()


class Token():
    token_type: TokenType
    literal: str

    def __init__(self, token_type: TokenType, literal: str):
        self.token_type = token_type
        self.literal = literal


class Scanner():
    source_code: str
    current_index: int

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.current_index = 0

    def already_read_all_source_code(self) -> bool:
        return self.current_index >= len(self.source_code)

    def get_char_with_current_index(self) -> str:
        return self.source_code[self.current_index]

    def get_char_with_next_index(self) -> str:
        # if current index is equal or greter than source code length then return
        # because there is no string after that
        # ex) souce_code is 'abc', self.current_index is 2
        last_index_of_source_code = len(self.source_code) - 1
        if self.current_index >= last_index_of_source_code:
            return ''

        return self.source_code[self.current_index + 1]

    def advance_current_index(self) -> None:
        self.current_index += 1

    def is_digit(self, c: str) -> bool:
        return c in [str(n) for n in range(0,10,1)]

    def number_to_token(self, c: str) -> Token:
        current_token_start_index = self.current_index

        # Int value
        while self.get_char_with_next_index() != '':

            # 3+, 1-
            if self.is_digit(self.get_char_with_next_index()) is False:
                break

            self.advance_current_index()

        return Token(
            token_type=TokenType.NUMBER,
            literal=self.source_code[current_token_start_index:self.current_index + 1]
        )

    def scan(self) -> List[Token]:
        res: List[Token] = []

        while self.already_read_all_source_code() is False:
            c: str = self.get_char_with_current_index();

            if c == '+':
                res.append(Token(token_type=TokenType.PLUS, literal='+'))
            elif c == '-':
                res.append(Token(token_type=TokenType.MINUS, literal='-'))
            elif self.is_digit(c):
                res.append(self.number_to_token(c))

            if self.already_read_all_source_code() is False:
                self.advance_current_index()

        return res

class AST():
    pass


class Number(AST):
    value: int

    def __init__(self, token: Token):
        if token.token_type != TokenType.NUMBER:
            raise ValueError(f'type is not NUMBER: {token.token_type}')

        self.value = int(token.literal)


class BinaryOperator(AST):
    left: AST
    operator: TokenType
    right: AST

    def __init__(self, left: AST, token_type: TokenType, right: AST):
        self.left = left
        self.operator = token_type
        self.right = right


class Parser():
    tokens: List[Token]
    current_index: int

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current_index = 0

    def at_end(self) -> bool:
        last_index_of_tokens = len(self.tokens) - 1
        return self.current_index >= last_index_of_tokens

    def previous_token(self) -> Token:
        if self.current_index <= 0:
            raise RuntimeError('current index is 0, but you are trying to read previous token')

        return self.tokens[self.current_index - 1]

    def next_token(self) -> Token:
        last_index_of_tokens = len(self.tokens) - 1
        if self.current_index >= last_index_of_tokens:
            raise RuntimeError('current index is last index of tokens, but you are trying to read next token')

        return self.tokens[self.current_index + 1]

    def current_token(self) -> Token:
        return self.tokens[self.current_index]

    def advance_current_index(self) -> None:
        self.current_index += 1
    
    def is_binary_operator_token(self, token: Token) -> bool:
        return token.token_type in [TokenType.PLUS, TokenType.MINUS]
    
    def is_number_token(self, token: Token) -> bool:
        return token.token_type is TokenType.NUMBER

    def parse(self) -> AST:
        ast: List[AST] = []

        while self.at_end() is False:

            if self.is_binary_operator_token(self.current_token()):
                ast.append(
                    BinaryOperator(
                        left=Number(token=self.previous_token()),
                        token_type=self.current_token().token_type,
                        right=self.search_next_ast()
                    )
                )
            elif self.is_number_token(self.current_token()):
                pass

            self.advance_current_index()
        return ast[0]

    def search_next_ast(self) -> AST:
        ast: List[AST] = []

        # initialize with right side of BinaryOperator
        # + 2 <- 2 is the token
        self.advance_current_index()
        ast.append(
            Number(
                self.current_token()
            )
        )

        while self.at_end() is False:

            if self.is_binary_operator_token(self.current_token()):
                ast[0] = BinaryOperator(
                    left=Number(token=self.previous_token()),
                    token_type=self.current_token().token_type,
                    right=self.search_next_ast()
                )
                break
                
            elif self.is_number_token(self.current_token()):
                pass

            self.advance_current_index()
        return ast[0]

class Interpreter():
    ast: AST
    current_ast: AST
    op_array: List[AST]

    def __init__(self, ast: AST):
        self.ast = ast
        self.current_ast = ast
    
    def resolved(self) -> bool:
        if isinstance(self.current_ast, BinaryOperator) is False:
            raise RuntimeError('current ast is not BinaryOperator')
        return isinstance(self.current_ast.right, Number)
    
    def create_op_array(self) -> List[AST]:
        res: List[AST] = []
        while self.resolved() is False:
            res.append(self.current_ast)
            self.current_ast = self.current_ast.right
        
        # the last ast's right is Number
        # we have to add res to the last ast but while did not append it
        res.append(self.current_ast)
        return res 
    
    def calc(self, left: Number, op: TokenType, right: Number) -> Number:
        if op is TokenType.PLUS:
            token = Token(
                TokenType.NUMBER,
                literal=f'{left.value + right.value}'
            )
            return Number(token)
        elif op is TokenType.MINUS:
            token = Token(
                TokenType.NUMBER,
                literal=f'{left.value - right.value}'
            )
            return Number(token)
        else:
            raise RuntimeError('unexpected token')
    
    def interpret(self) -> None:
        self.op_array = self.create_op_array()

        while len(self.op_array) > 1:
            res = self.calc(self.op_array[-1].left, self.op_array[-1].operator, self.op_array[-1].right)
            self.op_array.pop()
            self.op_array[-1].right = res
        
        res = self.calc(self.op_array[-1].left, self.op_array[-1].operator, self.op_array[-1].right)
        print(res.value)

def main(source_code: str) -> None:
    scanner = Scanner(source_code)
    tokens = scanner.scan()
    parser = Parser(tokens)
    ast: AST = parser.parse()
    interpreter = Interpreter(ast)
    interpreter.interpret()


if __name__ == '__main__':
    main('1+2+3') # 6
    main('1+2-3') # 0
    main('12-4') # 8
