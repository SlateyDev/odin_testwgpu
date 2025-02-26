package test

plane_vertex_data := []Vertex {
	vertex(-1, -1, 0, 0, 1, 1, 1, 1),
	vertex(-1, 1, 0, 0, 0, 1, 1, 1),
	vertex(1, 1, 0, 1, 0, 1, 1, 1),
	vertex(1, -1, 0, 1, 1, 1, 1, 1),
}

plane_index_data: []u32 = {
	0, 1, 2, 2, 3, 0,
}