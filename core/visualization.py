try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    Digraph = None


def trace(root):
    """
    Builds a set of all nodes and edges in a computational graph.
    
    Args:
        root: The root Value node of the computational graph
        
    Returns:
        tuple: (nodes_set, edges_set) representing the computational graph
    """
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, format='svg', rankdir='LR'):
    """
    Draws the computational graph of a Value using graphviz.
    
    Args:
        root: The root Value node to visualize
        format (str): Output format ('svg', 'png', etc.)
        rankdir (str): Graph direction ('LR' for left-to-right, 'TB' for top-to-bottom)
        
    Returns:
        Digraph: The graphviz Digraph object, or None if graphviz is not available
        
    Note:
        Requires graphviz to be installed: pip install graphviz
    """
    if not HAS_GRAPHVIZ:
        print("Warning: graphviz not available. Install with: pip install graphviz")
        return None
    
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    for n in nodes:
        # Create main node
        dot.node(
            name=str(id(n)), 
            label="{%s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), 
            shape='record'
        )
        
        # Create operation node if there's an operation
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    # Add edges between nodes
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
