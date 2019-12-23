import ast
import llvmlite.ir as IR

class Module(ast.AST):
  '''
  AST中的根元素, body为一系列AAST节点
  '''
  _fields = ['body']
  def __init__(self, body: list):
    self.body = body

class Variable(ast.AST):
  '''
  变量，来自ast.Name
  '''
  _fields = ['varname', 'type']
  def __init__(self, _id, _type=None):
    self.id = _id
    self.type = _type

class Assign(ast.AST):
  '''
  Assign节点
  target：ast.Name->Variable, ast.Subscript->Subscript
  value: ast.Name->Variable, ast.Subscript->Subscript, ast.BinOP->BinOp
  '''
  _fields = ['target', 'value', 'type']
  def __init__(self, target, value, _type=None):
    self.target = target
    self.value = value
    if value.type:
      self.type = value.type
    self.type = _type

class Constant(ast.AST):
  '''
  常量，如True和False
  '''
  _fields = ['value', 'type']
  def __init__(self, value):
    self.value = value
    self.type = type(self.value).__name__
      
class Return(ast.AST):
  '''
  Return节点，只支持返回Name类型
  varname: ast.Name.id
  '''
  _fields = ['varname']
  def __init__(self, varname: str):
    self.varname = varname

class Subscript(ast.AST):
  '''
  Subscript节点，如s[i]或s[1]
  varname: 上文中的's'
  index: 上文中的'i'或'1' | ast.Name->Variable, ast.Num->PyInt
  '''
  _fields = ['varname', 'index', 'type']
  def __init__(self, varname: str, index: int, _type='int'):
    self.varname = varname
    self.index = index
    self.type = _type

class Function(ast.AST):
  '''
  Function节点
  name：函数名
  args：参数表 | [(参数名，参数类型), (参数名，参数类型), ...]
  retType: 返回值类型 | 'int', 'bool', 'ptr'
  body: 函数体 | [AAST节点, AAST节点, ...]
  '''
  _fields = ['name', 'args', 'retType', 'body']
  def __init__(self, name: str, args: list, body, retType: str):
    self.name = name
    self.args = args
    self.retType = retType
    self.body = body

class PyInt(ast.AST):
  '''
  PyInt节点，来自ast.Num，只支持int类型
  '''
  _fields = ['n']
  def __init__(self, n, _type=None):
    self.n = n
    self.type = 'int'

class BinOp(ast.AST):
  '''
  二元操作，包括 + - * /
  left/right: Variable, PyInt, Constant
  op: string | '+', '-', '*', '/'
  '''
  _fields = ['left', 'right', 'op', 'type']
  def __init__(self, left, right, op, _type=None):
    # 不支持类型转换
    if left.type:
      self.type = left.type
      if right.type and self.type != left.type:
        raise TypeError
    elif right.type:
      self.type = right.type
    else:
      self.type = _type
    self.left = left
    self.right = right
    self.op = op

class Compare(ast.AST):
  '''
  比较节点
  left/right: Variable, PyInt, Constant
  op: string | '==', '!=', '&&', '||'
  '''
  _fields = ['left', 'right', 'op', 'type']
  def __init__(self, left, right, op):
    self.type = 'bool'
    self.right = right
    self.left = left
    self.op = op

class While(ast.AST):
  '''
  While循环节点
  test: 循环条件 | Constant, Compare
  body: 循环体 | [AAST节点, AAST节点, ...]
  '''  
  _fields = ['test', 'body']
  def __init__(self, test, body):
    self.test = test
    self.body = body

class Ifelse(ast.AST):
  '''
  If-else节点
  test: IF-block执行的条件 |　Constant, Compare
  ifbody: if-block | [AAST节点, AAST节点, ...]
  elbody: el-block | [AAST节点, AAST节点, ...]
  '''
  _fields = ['test', 'ifbody', 'elbody']
  def __init__(self, test, ifbody, elbody):
    self.test = test
    self.ifbody = ifbody
    self.elbody = elbody

class List(ast.AST):
  '''
  List节点
  content: list的元素 | [PyInt, PyInt, ...]
  type: 元素的类型
  length: 长度
  '''
  _fields = ['content', 'length', 'type']
  def __init__(self, content, _type):
    self.content = content
    self.length = len(self.content)
    self.type = _type

class Call(ast.AST):
  '''
  Call节点
  func: function的name
  args: 参数表 | [PyInt, Variable, ...]
  '''
  _fields = ['func', 'args', 'type']
  def __init__(self, func, args, _type=None):
    self.func = func
    self.args = args
    self.type = _type

class Break(ast.AST):
  _fields = []
  def __init__(self):
    pass

PyTypeTable = {
  'int': 'int32',
  'bool': 'bool',
  'str': 'str'
}

def getPyOp(op):
  if isinstance(op, ast.Add):
    return '+'
  if isinstance(op, ast.Sub):
    return '-'
  if isinstance(op, ast.Mult):
    return '*'
  if isinstance(op, ast.Div):
    return '/'
  if isinstance(op, ast.Eq):
    return '=='
  if isinstance(op, ast.NotEq):
    return '!='
  if isinstance(op, ast.And):
    return '&&'
  if isinstance(op, ast.Or):
    return '||'
  if isinstance(op, ast.Lt):
    return '<'
  if isinstance(op, ast.LtE):
    return '<='
  if isinstance(op, ast.Gt):
    return '>'
  if isinstance(op, ast.GtE):
    return '>='
  raise NotImplementedError

class PyVistor(ast.NodeVisitor):
  def __init__(self):
    pass

  def __call__(self, source):
    self._source = source
    self._ast = ast.parse(source)
    return self.visit(self._ast)
  
  def visit_Break(self, node: ast.Break):
    return Break()
    
  def visit_Module(self, node: ast.Module):
    body = [self.visit(item) for item in node.body]
    return Module(body)

  def visit_FunctionDef(self, node: ast.FunctionDef):
    args = [(item.arg, PyTypeTable[item.annotation.id]) for item in node.args.args]
    name = node.name
    body = [self.visit(item) for item in node.body] 
    ret = PyTypeTable[node.returns.id]
    return Function(name, args, body, ret)

  def visit_Name(self, node: ast.Name):
    return Variable(node.id)

  def visit_NameConstant(self, node: ast.NameConstant):
    return Constant(node.value)

  def visit_Subscript(self, node: ast.Subscript):
    varname = node.value.id
    index = self.visit(node.slice.value)
    return Subscript(varname, index)
    # need try-except

  def visit_Num(self, node: ast.Num):
    return PyInt(node.n)

  def visit_Assign(self, node: ast.Assign):
    if len(node.targets) > 1:
      raise NotImplementedError
    target = self.visit(node.targets[0])          # Get PyInt or Subscript or Variable
    value = self.visit(node.value)
    return Assign(target, value)    

  def visit_BinOp(self, node: ast.BinOp):
    op = getPyOp(node.op)
    left = self.visit(node.left)
    right = self.visit(node.right)
    return BinOp(left, right, op)
  
  def visit_Compare(self, node: ast.Compare):
    if len(node.ops) > 1:
      raise NotImplementedError
    op = getPyOp(node.ops[0])
    left = self.visit(node.left)
    if len(node.comparators) > 1:
      raise NotImplementedError
    right = self.visit(node.comparators[0])
    return Compare(left, right, op)

  def visit_While(self, node: ast.While):
    test = self.visit(node.test)
    body = [self.visit(item) for item in node.body]
    return While(test, body)

  def visit_If(self, node: ast.If):
    test = self.visit(node.test)
    ifbody = [self.visit(item) for item in node.body]
    elbody = [self.visit(item) for item in node.orelse]
    return Ifelse(test, ifbody, elbody)

  def visit_List(self, node: ast.List):
    content = []
    for item in node.elts:
      content.append(item.n)
    return List(content, 'list')

  def visit_Call(self, node: ast.Call):
    func = node.func.id   # support calling names only
    args = [self.visit(item) for item in node.args]
    return Call(func, args)

  def visit_Str(self, node: ast.Str):
    content = list(bytes(node.s, encoding='utf-8'))
    return List(content, 'str')
    
  def visit_Return(self, node: ast.Return):
    return Return(node.value.id)

llvmTypeTable = {
  'int32': IR.IntType(32),
  'int': IR.IntType(32),
  'bool': IR.IntType(1),
  'str': IR.IntType(32),
  None: IR.IntType(1)
}

def getUniqueName(name, other):
  return name

class LLVMGen():
  def __init__(self, module):
    self.function = None
    self.locals = {}
    self.builder = None
    self.exit_block = None
    self.module = module
    self.functions = {}
    self.blocks = {}
    self.cur_while_index = -1
    
  def start_function(self, name, retType, args):
    if self.function != None:
      raise SyntaxError
    funcType = IR.FunctionType(retType, args)
    function = IR.Function(self.module, funcType, name)
    self.entry_block = function.append_basic_block('entry')
    builder = IR.builder.IRBuilder(self.entry_block)
    self.exit_block = function.append_basic_block('exit')
    self.function = function
    self.builder = builder
    self.blocks['entry'] = self.entry_block
    self.blocks['exit'] = self.exit_block

  def end_function(self):
    self.builder.branch(self.exit_block)
    self.builder.position_at_end(self.exit_block)
    if 'retval' in self.locals:
      retval = self.builder.load(self.locals['retval'])
      self.builder.ret(retval)
    else:
      self.builder.ret_void()
    # clean up local variables
    # todo: 清理block缓存
    newLocals = {}
    for (key, value) in self.locals.items():
      if not (str(key).startswith(self.function.name + '$') or str(key) == 'retval'):
        newLocals.setdefault(key, value)
    self.locals = newLocals
    self.functions.setdefault(self.function.name, (self.function, self.builder))
    self.function = None
    self.builder = None
    self.exit_block = None

  def visit_Assign(self, node: Assign):
    value = self.visit(node.value)
    addr = self.visit(node.target, type=llvmTypeTable[node.value.type])
    if isinstance(node.value, List):
      name = self.function.name + '$' + node.target.id
      self.locals[name] = value
    else:
      self.builder.store(value, addr)
    
  def visit_Break(self, node: Break):
    self.builder.branch(self.blocks["while_block_" + str(self.cur_while_index) + ".endif"])
    #self.builder.branch(self.blocks[])

  def visit_BinOp(self, node: BinOp):
    left = self.visit(node.left)
    right = self.visit(node.right)

    if isinstance(left, IR.AllocaInstr):
      left_ = self.builder.load(left)
    else:
      left_ = left

    if isinstance(right, IR.AllocaInstr):
      right_ = self.builder.load(right)
    else:
      right_ = right

    if node.op == '+':
      return self.builder.add(left_, right_)
    elif node.op == '-':
      return self.builder.sub(left_, right_)
    elif node.op == '*':
      return self.builder.mul(left_, right_)
    elif node.op == '/':
      return self.builder.sdiv(left_, right_)

  def visit_Module(self, node: Module):
    body = [self.visit(item) for item in node.body]
  
  def visit_PyInt(self, node: PyInt):
    return llvmTypeTable['int32'](node.n)

  def visit_int(self, node: int):
    return llvmTypeTable['int32'](node)

  def visit_Constant(self, node: Constant):
    return llvmTypeTable['bool'](node.value)

  def visit_Variable(self, node: Variable, type=None):
    if node == None:
      return
    name = self.function.name + '$' + node.id
    if name in self.locals:
      addr = self.locals[name]
      return addr
    else:
      if type != None:
        node.type = type
      else:
        node.type = IR.IntType(32)
      
      alloca=self.builder.alloca(
            typ=node.type,
            name=name)
      self.locals[name] = alloca
      return alloca

  def visit_List(self, node: List):
    alloca = self.builder.alloca(llvmTypeTable[node.type], size=(node.length + 20))

    for i in range(node.length):
      self.builder.store(self.visit(node.content[i]), alloca, align=i)

    return alloca

  def visit_Function(self, node: Function):
    retType = node.retType
    args = tuple([llvmTypeTable[item[1]] for item in node.args]) #type
    name = getUniqueName(node.name, args)
    self.start_function(name, llvmTypeTable[retType], args)
    for i in range(len(self.function.args)):
      self.function.args[i].name = node.args[i][0]
    for item in self.function.args:
      mem = self.builder.alloca(item.type)
      self.builder.store(item, mem)
      self.locals.setdefault(name + '$' + item.name, mem)
    if retType != 'void':
      self.locals['retval'] = self.builder.alloca(llvmTypeTable[retType], name='retval')
    for item in node.body:
      self.visit(item)
    self.end_function()

  def visit_Subscript(self, node: Subscript):
    name = self.function.name + '$' + node.varname
    if name in self.locals:
      if isinstance(node.index, Variable):
        index_name = self.function.name + '$' + node.index.id
        a = self.locals[index_name]
        b = self.builder.ptrtoint(a, IR.IntType(32))
        return self.builder.gep(self.locals[name], [b])
      else:
        index = self.visit(node.index)
        return self.builder.gep(self.locals[name], [index])
    else:
      raise TypeError
  
  def visit_Compare(self, node: Compare):
    return self.builder.icmp_signed(node.op, self.visit(node.left), self.visit(node.right))

  def visit_Ifelse(self, node: Ifelse):
    test = self.visit(node.test)
    with self.builder.if_else(test) as (then, otherwise):
      with then:
        for subnode in node.ifbody:
          self.visit(subnode)
      with otherwise:
        for subnode in node.elbody:
          self.visit(subnode)
    #self.builder.position_at_end(self.entry_block)

  def visit_While(self, node: While):
    body = node.body
    index = 0
    
    while 'while_block_' + str(index) in self.blocks:
      index += 1
    
    self.cur_while_index = index

    block = self.builder.append_basic_block('while_block_' + str(index))
    self.blocks['while_block_' + str(index)] = block
    self.builder.position_at_end(block)

    with self.builder.if_then(self.visit(node.test)) as end_block:
      self.blocks['while_block_' + str(index) + '.endif'] = end_block
      for subnode in node.body:
        self.visit(subnode)
      self.builder.branch(block)

  def visit_Return(self, node: Return):
    # can only return actual value
    if 'retval' in self.locals:
      value = self.locals[self.function.name + '$' + node.varname]
      val = self.builder.load(value)
      self.builder.store(val, self.locals['retval'])
    else:
      self.builder.branch(self.exit_block)

  def visit(self, node: ast.AST, type=None):
    if node == None:
      return
    if isinstance(node, Variable):
      return self.visit_Variable(node, type)
    elif isinstance(node, Subscript):
      return self.visit_Subscript(node)
    name = 'visit_' + node.__class__.__name__
    if name == 'visit_NoneType':
      return
    if hasattr(self, name):
      return getattr(self, name)(node)
    else:
      raise NotImplementedError(name)

module = IR.Module('test')
cgen = LLVMGen(module)

with open('source.py') as f:
  ast_v = PyVistor()
  tree = ast_v(f.read())
  cgen.visit(tree)
  with open('source.ll', 'w') as f:
    f.write(str(module))
