#+feature dynamic-literals
package test

foreign import customImports "customImports"

import "core:sys/wasm/js"
import "core:unicode"
import "core:unicode/utf8"

import "vendor:wgpu"
import mu "vendor:microui"
import "core:image"
import "core:fmt"
// import "vendor:cgltf"

OS :: struct {
	initialized: bool,
    clipboard: [dynamic]byte,
	keyStates: map[u32]bool,
}

@(private="file")
KEY_CODE_MAP := map[string]u32{
    "KeyW" = 87,
    "KeyA" = 65,
    "KeyS" = 83,
    "KeyD" = 68,
}

os : OS
mu_state := struct {
	cursor: [2]i32,
}{}

os_init :: proc() {
	context = state.ctx

	os.keyStates = make(map[u32]bool, context.allocator)

	assert(js.add_window_event_listener(.Key_Down, nil, key_down_callback))
	assert(js.add_window_event_listener(.Key_Up, nil, key_up_callback))
	assert(js.add_window_event_listener(.Mouse_Down, nil, mouse_down_callback))
	assert(js.add_window_event_listener(.Mouse_Up, nil, mouse_up_callback))
	assert(js.add_event_listener("wgpu-canvas", .Mouse_Move, nil, mouse_move_callback))
	assert(js.add_window_event_listener(.Wheel, nil, scroll_callback))
	assert(js.add_window_event_listener(.Resize, nil, size_callback))
}

// NOTE: frame loop is done by the runtime.js repeatedly calling `step`.
os_run :: proc() {
	os.initialized = true
}

@(private="file", export)
step :: proc(dt: f32) -> bool {
	context = state.ctx

    if !os.initialized {
		return true
	}
    
	if captured_mouse {
		forward := (os.keyStates[KEY_CODE_MAP["KeyW"]] ? 1 : 0) + (os.keyStates[KEY_CODE_MAP["KeyS"]] ? -1 : 0)
		right := (os.keyStates[KEY_CODE_MAP["KeyD"]] ? 1 : 0) + (os.keyStates[KEY_CODE_MAP["KeyA"]] ? -1 : 0)
		camera_move_forward(&flyCamera, f32(forward) * 10 * dt)
		camera_move_right(&flyCamera, f32(right) * 10 * dt)
	}

	frame(dt)

	free_all(context.temp_allocator)
	return true
}

os_get_render_bounds :: proc() -> (width, height: u32) {
	rect := js.get_bounding_client_rect("body")
	dpi := os_get_dpi()
	return u32(f32(rect.width) * dpi), u32(f32(rect.height) * dpi)
}

os_get_dpi :: proc() -> f32 {
	ratio := f32(js.device_pixel_ratio())
	return ratio
}

os_get_framebuffer_size :: proc() -> (width, height: u32) {
	rect := js.get_bounding_client_rect("body")
	dpi := js.device_pixel_ratio()
	return u32(f64(rect.width) * dpi), u32(f64(rect.height) * dpi)
}

os_get_surface :: proc(instance: wgpu.Instance) -> wgpu.Surface {
	return wgpu.InstanceCreateSurface(
		instance,
		&wgpu.SurfaceDescriptor{
			nextInChain = &wgpu.SurfaceSourceCanvasHTMLSelector{
				sType = .SurfaceSourceCanvasHTMLSelector,
				selector = "#wgpu-canvas",
			},
		},
	)
}

os_set_clipboard :: proc(_: rawptr, text: string) -> bool {
	// TODO: Use browser APIs
	clear(&os.clipboard)
	append(&os.clipboard, text)
	return true
}

os_get_clipboard :: proc(_: rawptr) -> (string, bool) {
	// TODO: Use browser APIs
	return string(os.clipboard[:]), true
}

@(private="file", fini)
os_fini :: proc "contextless" () {
	context = state.ctx

	delete(os.keyStates)

	js.remove_window_event_listener(.Key_Down, nil, key_down_callback)
	js.remove_window_event_listener(.Key_Up, nil, key_up_callback)
	js.remove_window_event_listener(.Mouse_Down, nil, mouse_down_callback)
	js.remove_window_event_listener(.Mouse_Up, nil, mouse_up_callback)
	js.remove_event_listener("wgpu-canvas", .Mouse_Move, nil, mouse_move_callback)
	js.remove_window_event_listener(.Wheel, nil, scroll_callback)
	js.remove_window_event_listener(.Resize,   nil, size_callback)

    finish()
}

@(private="file")
size_callback :: proc(e: js.Event) {
	resize()
}

@(private="file")
KEY_MAP := map[string]mu.Key{
	"ShiftLeft"    = .SHIFT,
	"ShiftRight"   = .SHIFT,
	"ControlLeft"  = .CTRL,
	"ControlRight" = .CTRL,
	"MetaLeft"     = .CTRL,
	"MetaRight"    = .CTRL,
	"AltLeft"      = .ALT,
	"AltRight"     = .ALT,
	"Backspace"    = .BACKSPACE,
	"Delete"       = .DELETE,
	"Enter"        = .RETURN,
	"ArrowLeft"    = .LEFT,
	"ArrowRight"   = .RIGHT,
	"Home"         = .HOME,
	"End"          = .END,
	"KeyA"         = .A,
	"KeyX"         = .X,
	"KeyC"         = .C,
	"KeyV"         = .V,
}

@(private="file")
key_down_callback :: proc(e: js.Event) {
	context = state.ctx

	js.event_prevent_default()

	if k, ok := KEY_CODE_MAP[e.data.key.code]; ok {
		os.keyStates[k] = true
	}

	if k, ok := KEY_MAP[e.data.key.code]; ok {
		mu.input_key_down(&mu_ctx, k)
	}

	if .CTRL in mu_ctx.key_down_bits {
		return
	}

	ch, size := utf8.decode_rune(e.data.key.key)
	if len(e.data.key.key) == size && unicode.is_print(ch) {
		mu.input_text(&mu_ctx, e.data.key.key)
	}
}

@(private="file")
key_up_callback :: proc(e: js.Event) {
	context = state.ctx

	if k, ok := KEY_CODE_MAP[e.data.key.code]; ok {
		os.keyStates[k] = false
	}

	if k, ok := KEY_MAP[e.data.key.code]; ok {
		mu.input_key_up(&mu_ctx, k)
	}

	js.event_prevent_default()
}

captured_mouse := false

@(private="file")
mouse_down_callback :: proc(e: js.Event) {
	context = state.ctx

	switch e.data.mouse.button {
	case 0: mu.input_mouse_down(&mu_ctx, mu_state.cursor.x, mu_state.cursor.y, .LEFT)
	case 1: mu.input_mouse_down(&mu_ctx, mu_state.cursor.x, mu_state.cursor.y, .MIDDLE)
	case 2: mu.input_mouse_down(&mu_ctx, mu_state.cursor.x, mu_state.cursor.y, .RIGHT)
		if !captured_mouse {
			web_request_pointer_lock("wgpu-canvas", 11)
			captured_mouse = true
		}
	}

	js.event_prevent_default()
}

@(private="file")
mouse_up_callback :: proc(e: js.Event) {
	context = state.ctx

	switch e.data.mouse.button {
	case 0: mu.input_mouse_up(&mu_ctx, mu_state.cursor.x, mu_state.cursor.y, .LEFT)
	case 1: mu.input_mouse_up(&mu_ctx, mu_state.cursor.x, mu_state.cursor.y, .MIDDLE)
	case 2: mu.input_mouse_up(&mu_ctx, mu_state.cursor.x, mu_state.cursor.y, .RIGHT)
		if captured_mouse {
			web_release_pointer_lock()
			captured_mouse = false
		}
	}

	js.event_prevent_default()
}

@(private="file")
mouse_move_callback :: proc(e: js.Event) {
	context = state.ctx
	mu_state.cursor = {i32(e.data.mouse.offset.x), i32(e.data.mouse.offset.y)}
	mu.input_mouse_move(&mu_ctx, mu_state.cursor.x, mu_state.cursor.y)

	if captured_mouse {
		camera_adjust_pitch(&flyCamera, f32(e.data.mouse.movement.y))
		camera_adjust_yaw(&flyCamera, f32(e.data.mouse.movement.x))
	}
}

@(private="file")
scroll_callback :: proc(e: js.Event) {
	context = state.ctx
	mu.input_scroll(&mu_ctx, i32(e.data.wheel.delta.x), i32(e.data.wheel.delta.y))
}

COMPTIME_ASSETS := #load_directory("./assets")

os_load_image :: proc(path: string) -> (output: ^image.Image, err: image.Error) {
	for file in COMPTIME_ASSETS {
		if file.name == path {
			output, err = image.load_from_bytes(file.data)
			return
		}
	}

	fmt.eprintfln("ERROR: Asset not found [%s]", path)

	return nil, .Unable_To_Read_File
}


// os_load_gltf :: proc(path: cstring) -> (output: Mesh) {
// 	for file in COMPTIME_ASSETS {
// 		if file.name == string(path) {
// 			return load_gltf_from_bytes(file.data)
// 		}
// 	}

// 	fmt.eprintfln("ERROR: Asset not found [%s]", path)
// 	return
// }

@(default_calling_convention="contextless")
foreign customImports {
	web_request_pointer_lock           :: proc(canvasIdPtr: string, canvasIdLen: u32) ---
	web_release_pointer_lock           :: proc() ---
}