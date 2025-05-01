package test

import "vendor:wgpu"

EngineTexture :: struct {
    texture: wgpu.Texture,
    view: wgpu.TextureView,

    from_colour_f32: proc(engineTexture: ^EngineTexture, colour: [4]f32) -> ^EngineTexture,
    from_colour_u8: proc(engineTexture: ^EngineTexture, colour: [4]u8) -> ^EngineTexture,
    create_view: proc(engineTexture: ^EngineTexture) -> wgpu.TextureView,
}

new_EngineTexture :: proc() -> ^EngineTexture {
    from_colour_f32 :: proc(engineTexture: ^EngineTexture, colour: [4]f32) -> ^EngineTexture {
        return from_colour_u8(engineTexture, {u8(colour.r * 255), u8(colour.g * 255), u8(colour.b * 255), u8(colour.a * 255)})
    }
    
    from_colour_u8 :: proc(engineTexture: ^EngineTexture, colour: [4]u8) -> ^EngineTexture {
        colour := colour
        newTexture := wgpu.DeviceCreateTexture(state.device, &wgpu.TextureDescriptor{
            label = "Colour Texture",
            size = wgpu.Extent3D{1, 1, 1},
            usage = {.TextureBinding, .CopyDst},
            mipLevelCount = 1,
            sampleCount = 1,
            dimension = ._2D,
            format = .RGBA8Unorm,
        })
        wgpu.QueueWriteTexture(
            state.queue,
            &wgpu.TexelCopyTextureInfo{
                texture = newTexture,
                mipLevel = 0,
                origin = {},
                aspect = .All,
            },
            &colour,
            len(colour),
            &wgpu.TexelCopyBufferLayout{
                offset       = 0,
                bytesPerRow  = texture_format_bytes_per_row(.RGBA8Unorm, 1),
                rowsPerImage = 1,
            },
            &wgpu.Extent3D{
                width = 1,
                height = 1,
                depthOrArrayLayers = 1,
            },
        )

        engineTexture.texture = newTexture

        return engineTexture
    }

    create_view :: proc(engineTexture: ^EngineTexture) -> wgpu.TextureView {
        engineTexture.view = wgpu.TextureCreateView(engineTexture.texture)
        return engineTexture.view
    }

    return new_clone(EngineTexture{
        from_colour_f32 = from_colour_f32,
        from_colour_u8 = from_colour_u8,
        create_view = create_view,
    })
}

free_EngineTexture :: proc(engineTexture: ^EngineTexture) {
    if engineTexture.texture != nil {
        wgpu.TextureRelease(engineTexture.texture)
    }
    engineTexture.texture = nil
    if engineTexture.view != nil {
        wgpu.TextureViewRelease(engineTexture.view)
    }
    engineTexture.view = nil
    free(engineTexture)
}