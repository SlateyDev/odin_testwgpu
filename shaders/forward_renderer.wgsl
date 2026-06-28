struct Camera {
    view_proj: mat4x4<f32>,
    pos: vec4<f32>,
    forward: vec4<f32>,
}

struct CascadeInfo {
    view_proj: mat4x4<f32>,
    split_depth: f32,
}

struct Light {
    cascades: array<CascadeInfo, 4>,
    pos: vec4<f32>,
}

struct ModelMatrices {
    model_matrix: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> light: Light;
@group(1) @binding(0) var<uniform> model_matrices: ModelMatrices;

struct VertexInput {
    @builtin(vertex_index) in_vertex_index: u32,
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
    out.world_normal = normalize((model_matrices.normal_matrix * vec4<f32>(model.normal, 0.0)).xyz);

    var world_position: vec4<f32> = model_matrices.model_matrix * vec4<f32>(model.pos, 1.0);
    out.world_position = world_position.xyz;
    out.position = camera.view_proj * world_position;
    return out;
}

@group(2) @binding(0) var myTexture: texture_2d<f32>;
@group(2) @binding(1) var mySampler: sampler;

@group(3) @binding(0) var shadowMap: texture_depth_2d_array;
@group(3) @binding(1) var shadowSampler: sampler_comparison;

fn get_cascade_index(camera_space_z: f32) -> i32 {
    for (var i = 0; i < 4; i++) {
        if (camera_space_z < light.cascades[i].split_depth) {
            return i;
        }
    }
    return 4;
}

override shadowDepthTextureSize: f32 = 2048.0;

const cascade_colour_modulator: array<vec3<f32>, 4> = array<vec3<f32>, 4>(
    vec3<f32>(1.5, 0.5, 0.5),
    vec3<f32>(0.5, 1.5, 0.5),
    vec3<f32>(0.5, 0.5, 1.5),
    vec3<f32>(1.5, 0.5, 1.5)
);

@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    let light_dir = normalize(light.pos.xyz - in.world_position);
    let view_dir = normalize(camera.pos.xyz - in.world_position);
    let half_dir = normalize(view_dir + light_dir);

    let diffuse_strength = max(dot(in.world_normal, light_dir), 0.0);
    let diffuse_color = diffuse_strength;// * light.color;

    let specular_strength = pow(max(dot(in.world_normal, half_dir), 0.0), 32.0);
    let specular_color = specular_strength;// * light.color;

    let object_color = textureSample(myTexture, mySampler, in.tex_coords);

    let ambient_strength = 0.3;
    let ambient_color = ambient_strength;// * light.color;

    var result_color = (ambient_color + diffuse_color + specular_color) * object_color.xyz;

    let camera_to_fragment = in.world_position - camera.pos.xyz;
    let camera_depth = dot(camera_to_fragment, normalize(camera.forward.xyz));
    let cascade_idx = get_cascade_index(camera_depth);
    
    let shadowCoord = light.cascades[cascade_idx].view_proj * vec4<f32>(in.world_position, 1.0);
    let projCoords = shadowCoord.xyz / shadowCoord.w;
    let shadow_uv = projCoords.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    let shadow_depth = projCoords.z;

    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0 || shadow_depth < 0.0 || shadow_depth > 1.0 || cascade_idx >= 4) {
        return vec4<f32>(result_color, object_color.a);
    }

    let kernelSize: i32 = 1;
    let weightTotal: f32 = f32(kernelSize * 2 + 1) * f32(kernelSize * 2 + 1);

    // Percentage-closer filtering. Sample texels in the region to smooth the result.
    var visibility = 0.0;
    let oneOverShadowDepthTextureSize = 1.0 / shadowDepthTextureSize;
    for (var y = -kernelSize; y <= kernelSize; y++) {
        for (var x = -kernelSize; x <= kernelSize; x++) {
            let offset = vec2<f32>(vec2(x, y)) * oneOverShadowDepthTextureSize;
            visibility += textureSampleCompareLevel(
                shadowMap, 
                shadowSampler,
                shadow_uv + offset,
                cascade_idx,  // Layer index
                shadow_depth - (0.001 + 0.001 * f32(cascade_idx))
            );
        }
    }
    visibility /= weightTotal;

    let lambertian_factor = max(dot(light_dir, in.world_normal), 0.0);
    let lighting_factor = min(ambient_color + visibility * lambertian_factor, 1.0);

    return vec4<f32>(lighting_factor * result_color * cascade_colour_modulator[cascade_idx], object_color.a);
}
