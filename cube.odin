package test

vertex :: proc(x, y, z, u, v: f32, r : f32 = 1.0, g : f32 = 1.0, b : f32 = 1.0, a : f32 = 1.0) -> Vertex {
	return Vertex{
		position = {x, y, z},
		uv = {u, v},
        color = {r, g, b, a},
	}
}

cube_vertex_data := []Vertex {
	// top (0, 0, 1)
	vertex(-1, -1, 1, 0, 0),
	vertex(1, -1, 1, 1, 0),
	vertex(1, 1, 1, 1, 1),
	vertex(-1, 1, 1, 0, 1),
	// bottom (0, 0, -1)
	vertex(-1, 1, -1, 1, 0),
	vertex(1, 1, -1, 0, 0),
	vertex(1, -1, -1, 0, 1),
	vertex(-1, -1, -1, 1, 1),
	// right (1, 0, 0)
	vertex(1, -1, -1, 0, 0),
	vertex(1, 1, -1, 1, 0),
	vertex(1, 1, 1, 1, 1),
	vertex(1, -1, 1, 0, 1),
	// left (-1, 0, 0)
	vertex(-1, -1, 1, 1, 0),
	vertex(-1, 1, 1, 0, 0),
	vertex(-1, 1, -1, 0, 1),
	vertex(-1, -1, -1, 1, 1),
	// front (0, 1, 0)
	vertex(1, 1, -1, 1, 0),
	vertex(-1, 1, -1, 0, 0),
	vertex(-1, 1, 1, 0, 1),
	vertex(1, 1, 1, 1, 1),
	// back (0, -1, 0)
	vertex(1, -1, 1, 0, 0),
	vertex(-1, -1, 1, 1, 0),
	vertex(-1, -1, -1, 1, 1),
	vertex(1, -1, -1, 0, 1),
}

cube_index_data: []u16 = {
	0, 1, 2, 2, 3, 0, // top
	4, 5, 6, 6, 7, 4, // bottom
	8, 9, 10, 10, 11, 8, // right
	12, 13, 14, 14, 15, 12, // left
	16, 17, 18, 18, 19, 16, // front
	20, 21, 22, 22, 23, 20, // back
}