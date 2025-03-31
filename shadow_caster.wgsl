@group(0) @binding(0) var<uniform> mvp: mat4x4<f32>;

struct VertexInput {
    @location(0) pos: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) color0: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) proj_zw: vec2<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = mvp * model.pos;
    return out;
}

@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    var depth : f32 = in.position.z / in.position.w;
    return encode_depth(depth);
}

// // Karl noted this was from from https://aras-p.info/blog/2009/07/30/encoding-floats-to-rgba-the-final/
fn encode_depth(v: f32) -> vec4<f32> {
    var enc: vec4<f32> = vec4<f32>(1.0, 255.0, 65025.0, 16581375.0) * v;
    enc = fract(enc);
    enc -= enc.yzww * vec4<f32>(1.0/255.0, 1.0/255.0, 1.0/255.0, 0.0);
    return enc;
}