class Node:
    def __init__(self, attr, v):
        self.attribute = attr
        self.left = None
        self.right = None
        self.vote = v

def predict(node, example):
    if node.vote is not None:
        return node.vote
    else:
        if example[node.attribute] == 1:
            return predict(node.left, example)
        else:
            return predict(node.right, example)
        
        

root = Node('A', None)  
root.left = Node(None, 'Y=1')  
root.right = Node(None, 'Y=0') 

example = {'A': 1} 
result = predict(root, example) 
print(result)  