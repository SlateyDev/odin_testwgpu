struct Camera {
    view_proj: mat4x4<f32>,
    pos: vec4<f32>,
}

struct Light {
    view_proj: mat4x4<f32>,
    pos: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> light: Light;
@group(1) @binding(0) var<uniform> model_matrix: mat4x4<f32>;

struct VertexInput {
    @builtin(vertex_index) in_vertex_index: u32,
    @location(0) pos: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = light.view_proj * model_matrix * vec4<f32>(model.pos, 1.0);
    return out;
}
