@group(0) @binding(0) var<storage, read> mvp: array<mat4x4<f32>>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) instanceIndex: u32,
    @location(0) pos: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
) -> VertexOutput {
    var output: VertexOutput;
    output.position = mvp[instanceIndex] * pos;
    output.uv = uv;
    output.color = color;
    return output;
}

@group(1) @binding(0) var mySampler: sampler;
@group(1) @binding(1) var myTexture: texture_2d<f32>;

@fragment
fn fs_main(
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
) -> @location(0) vec4<f32> {
    let uv_color = textureSample(myTexture, mySampler, uv);
    // return vec4(linear_to_srgb(uv_color.rgb), uv_color.a) * color;
    // return vec4(srgb_to_linear(uv_color.rgb), uv_color.a) * color;
    return uv_color * color;
}

// Converts a linear (physical) color to sRGB space
fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let srgb_low = linear * 12.92;
    let srgb_high = pow(linear, vec3<f32>(1.0 / 2.4)) * 1.055 - 0.055;
    return select(srgb_high, srgb_low, linear <= cutoff);
}

// Converts an sRGB color to linear (physical) space
fn srgb_to_linear(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.04045);
    let linear_low = srgb / 12.92;
    let linear_high = pow((srgb + 0.055) / 1.055, vec3<f32>(2.4));
    return select(linear_high, linear_low, srgb <= cutoff);
}