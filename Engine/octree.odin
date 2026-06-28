package test

AABB :: struct {
    min, max: Vector3
}

Octree :: struct {
    root: ^OctreeNode
}

OctreeNode :: struct {
    bounds: AABB,
    modelIndices: [dynamic]int,
    children: [8]^OctreeNode,
    isLeaf: bool,
    depth: int,
}

GetModelAABB :: proc(model: ^Model) -> AABB {
    switch col in model.collider {
        case SphereCollider:
            r := col.radius * model.scale
            return MakeAABB(model.translation - Vector3{r, r, r}, model.translation + Vector3{r, r, r})
        case BoxCollider:
            axes := GetAxesFromRotationMatrix(model.rotationMatrix)
            half := col.size * model.scale
            rx := abs(axes[0].x)*half.x + abs(axes[1].x)*half.y + abs(axes[2].x)*half.z
            ry := abs(axes[0].y)*half.x + abs(axes[1].y)*half.y + abs(axes[2].y)*half.z
            rz := abs(axes[0].z)*half.x + abs(axes[1].z)*half.y + abs(axes[2].z)*half.z
            return MakeAABB(model.translation - Vector3{rx, ry, rz}, model.translation + Vector3{rx, ry, rz})
    }

    return {}

    MakeAABB :: proc(min, max: Vector3) -> AABB {
        return AABB{min, max}
    }
}

GetWorldAABB :: proc(models: []Model) -> AABB {
    result := GetModelAABB(&models[0])

    for i in 1..<len(models) {
        aabb := GetModelAABB(&models[i])
        result.min.x = min(result.min.x, aabb.min.x)
        result.min.y = min(result.min.y, aabb.min.y)
        result.min.z = min(result.min.z, aabb.min.z)
        result.max.x = max(result.max.x, aabb.max.x)
        result.max.y = max(result.max.y, aabb.max.y)
        result.max.z = max(result.max.z, aabb.max.z)
    }

    padding := Vector3{0.1, 0.1, 0.1}
    result.min -= padding
    result.max += padding

    return result
}

MakeOctree :: proc(bounds: AABB) -> Octree {
    return Octree{root = MakeOctreeNode(bounds, 0)}
}

MakeOctreeNode :: proc(bounds: AABB, depth: int) -> ^OctreeNode {
    node := new(OctreeNode)
    node.bounds = bounds
    node.isLeaf = true
    node.depth = depth
    node.modelIndices = make([dynamic]int)
    return node
}

OctreeInsert :: proc(octree: ^Octree, modelIdx: int, modelAABB: AABB, models: []Model) {
    InsertOctreeNode(octree.root, modelIdx, modelAABB, models)
}

InsertOctreeNode :: proc(node: ^OctreeNode, modelIds: int, modelAABB: AABB, models: []Model) {
    if !Overlaps(node.bounds, modelAABB) do return

    if node.isLeaf {
        if len(node.modelIndices) < OCTREE_MAX_OBJECTS || node.depth >= OCTREE_MAX_DEPTH {
            append(&node.modelIndices, modelIds)
            return
        }

        for i in 0..<8 {
            node.children[i] = MakeOctreeNode(GetOctantBounds(node.bounds, i), node.depth + 1)
        }
        node.isLeaf = false

        for existingIdx in node.modelIndices {
            existingAABB := GetModelAABB(&models[existingIdx])
            for i in 0..<8 {
                InsertOctreeNode(node.children[i], existingIdx, existingAABB, models)
            }
        }
        clear(&node.modelIndices)
    }

    for i in 0..<8 {
        InsertOctreeNode(node.children[i], modelIds, modelAABB, models)
    }

    Overlaps :: proc(a, b: AABB) -> bool {
        return a.min.x <= b.max.x && a.max.x >= b.min.x &&
               a.min.y <= b.max.y && a.max.y >= b.min.y &&
               a.min.z <= b.max.z && a.max.z >= b.min.z
    }

    GetOctantBounds :: proc(parent: AABB, octant: int) -> AABB {
        center := (parent.min + parent.max) * 0.5
        result: AABB
        result.min.x = center.x     if octant & 1 != 0 else parent.min.x
        result.max.x = parent.max.x if octant & 1 != 0 else center.x
        result.min.y = center.y     if octant & 2 != 0 else parent.min.y
        result.max.y = parent.max.y if octant & 2 != 0 else center.y
        result.min.z = center.z     if octant & 4 != 0 else parent.min.z
        result.max.z = parent.max.z if octant & 4 != 0 else center.z
        return result
    }
}

OctreeGetCandidatePairs :: proc(octree: ^Octree) -> [][2]int {
    seen := make(map[[2]int]bool)
    pairs := make([dynamic][2]int)
    CollectPairs(octree.root, &seen, &pairs)
    return pairs[:]
    
    CollectPairs :: proc(node: ^OctreeNode, seen: ^map[[2]int]bool, pairs: ^[dynamic][2]int) {
        if node == nil do return

        if node.isLeaf {
            indices := node.modelIndices[:]
            for i in 0..<len(indices) {
                for j in i + 1..<len(indices) {
                    a, b := indices[i], indices[j]
                    if a > b { a, b = b, a }
                    key := [2]int{a, b}
                    if key not_in seen {
                        seen[key] = true
                        append(pairs, key)
                    }
                }
            }
            return
        }

        for child in node.children {
            CollectPairs(child, seen, pairs)
        }
    }
}
