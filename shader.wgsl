struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

struct Camera {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> model_matrices: array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> light : Light;

struct VertexInput {
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) instanceIndex: u32,
    @location(0) pos: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
}

@vertex
fn vs_main(
    model : VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.world_normal = normalize(model_matrices[model.instanceIndex] * vec4<f32>(model.normal, 1.0)).xyz;
    var world_position: vec4<f32> = model_matrices[model.instanceIndex] * vec4<f32>(model.pos, 1.0);
    out.world_position = world_position.xyz;
    out.position = camera.view_proj * world_position;
    return out;
}

@group(1) @binding(0) var mySampler: sampler;
@group(1) @binding(1) var myTexture: texture_2d<f32>;

@group(2) @binding(0) var shadowMap: texture_depth_2d;
@group(2) @binding(1) var shadowSampler: sampler_comparison;
@group(2) @binding(2) var<uniform> lightSpaceMatrix: mat4x4<f32>;

fn calculate_shadow(coord: vec4<f32>) -> f32 {
    let projCoords = coord.xyz / coord.w;
    let shadowCoord = projCoords.xy * 0.5 + 0.5;
    let depth = projCoords.z - 0.005; // Bias to prevent shadow acne
    return textureSampleCompare(shadowMap, shadowSampler, shadowCoord, depth);
}

@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    let light_dir = normalize(light.position - in.world_position);

    let view_dir = normalize(camera.view_pos.xyz - in.world_position);
    let half_dir = normalize(view_dir + light_dir);

    let diffuse_strength = max(dot(in.world_normal, light_dir), 0.0);
    let diffuse_color = light.color * diffuse_strength;

    let specular_strength = pow(max(dot(in.world_normal, half_dir), 0.0), 32.0);
    let specular_color = specular_strength * light.color;

    let object_color = textureSample(myTexture, mySampler, in.tex_coords);

    let ambient_strength = 0.1;
    let ambient_color = light.color * ambient_strength;

    var result_color = (ambient_color + diffuse_color + specular_color) * object_color.xyz;
    // return vec4(linear_to_srgb(uv_color.rgb), uv_color.a) * in.color;
    // return vec4(srgb_to_linear(uv_color.rgb), uv_color.a) * in.color;

    // let shadowCoord = lightSpaceMatrix * vec4<f32>(in.world_position, 1.0);
    // let shadowFactor = calculate_shadow(shadowCoord);
    // result_color = mix(result_color, vec3<f32>(0.0, 0.0, 0.0), 1.0 - shadowFactor);

    return vec4<f32>(result_color, object_color.a);
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