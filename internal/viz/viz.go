package viz

import (
	"fmt"
	"os"

	"github.com/dominikbraun/graph"
	"github.com/dominikbraun/graph/draw"
	"golang.org/x/exp/constraints"

	"github.com/richiejp/micrograd/internal/grad"
)

const opOffset = 10000

type edge struct {
	From uint64
	To uint64
}

func uint64Hash(x uint64) uint64 {
	return x
}

func trace[T constraints.Float](root *grad.Value[T]) (map[uint64]*grad.Value[T], []edge) {
	nodes := make(map[uint64]*grad.Value[T])
	var edges []edge

	var build func (v *grad.Value[T])
	build = func (v *grad.Value[T]) {
		if  _, ok := nodes[v.ID()]; ok {
			return
		}

		nodes[v.ID()] = v
		for _, child := range v.Prev() {
			edges = append(edges, edge{
				From: child.ID(),
				To: v.ID(),
			})
			build(child)
		}
	}
	build(root)

	return nodes, edges
}

func Render[T constraints.Float](path string, v *grad.Value[T]) error {
	g := graph.New(uint64Hash, graph.Directed())

	nodes, edges := trace(v)

	for _, n := range nodes {
		g.AddVertex(n.ID(), 
			graph.VertexAttribute("label", fmt.Sprintf("{ %s | data %.4f | grad %.4f }", n.Label(), n.Data(), n.Grad())),
			graph.VertexAttribute("shape", "record"),
		)

		if n.Op() != grad.OpNil {
			g.AddVertex(n.ID() + opOffset, graph.VertexAttribute("label", string(n.Op())))
			g.AddEdge(n.ID() + opOffset, n.ID())
		}
	}

	for _, e := range edges {
		g.AddEdge(e.From, e.To + opOffset)
	}

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("Create %s: %w", path, err)
	}
	return draw.DOT(g, file, /* draw.GraphAttribute("rankdir", "LR") */)
}
