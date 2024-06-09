import math
import random
from evalWithAbc import calAigeval

def eval_fn(state):
    if state[-1]=='7':
        state=state[:-1]
    return calAigeval(state,logFile='./mytask2/log',nextState='./mytask2/aig')

class Node:
    def __init__(self, state, depth,parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0
        self.depth=depth
    
    def is_fully_expanded(self):
        return len(self.children) == 8 or self.state[-1]=='7'
    
    def best_child(self, exploration_weight=1.0):
        weights = [
            (child.score / child.visits) + exploration_weight * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[weights.index(max(weights))]

    def add_child(self, child_state):
        child = Node(state=child_state, parent=self,depth=self.depth+1)
        self.children.append(child)
        return child

class MCTS:
    def __init__(self, initial_state, eval_fn=eval_fn, iterations=1000, max_depth=10):
        self.root = Node(state=initial_state,depth=1)
        self.eval_fn = eval_fn
        self.iterations = iterations
        self.max_depth = max_depth
        for _ in range(iterations):
            self.bestaction=self.search()
    
    def select(self, node):
        while not node.is_fully_expanded() and node.children:
            node = node.best_child()
        return node
    
    def expand(self, node):
        if node.is_fully_expanded() or node.depth==self.max_depth:
            return node
        while True:
            new_state = self.get_random_state(node.state)
            if not new_state in [child.state for child in node.children]:
                break
        return node.add_child(new_state)
    
    def simulate(self, node):
        current_state = node.state
        for _ in range(self.max_depth-node.depth):
            if self.is_terminal(current_state):
                break
            current_state = self.get_random_state(current_state)
        return self.eval_fn(current_state)
    
    def backpropagate(self, node, score):
        while node is not None:
            node.visits += 1
            node.score += score
            node = node.parent
    
    def get_random_state(self, state):
        # 应根据问题定义生成一个新的随机状态
        new_state=state+str(random.choice(range(0,8)))
        return new_state  # 示例，实际需要根据实际情况修改
    
    def is_terminal(self, state):
        # 判断状态是否为终止状态
        return state[-1]=='7'  # 示例，实际需要根据实际情况修改
    
    def search(self):
        for _ in range(self.iterations):
            node = self.select(self.root)
            node = self.expand(node)
            score = self.simulate(node)
            self.backpropagate(node, score)
        return self.root.best_child(exploration_weight=0)  # 返回最优子节点

initial_state='adder_'
max_eval=calAigeval(initial_state)
max_index=0
state=[initial_state]
for i in range(10):
    state.append(MCTS(state[-1],max_depth=10-i,iterations=100).bestaction.state)
    val=calAigeval(state[-1])
    if state[-1][-1]=='7':
        state[-1]=state[-1][:-1]
        break
    print(state[-1],val)
    if val > max_eval:
        max_index=len(state)-1
        max_eval=val
print('Best Action is'+state[max_index])
