REM NOTE: changing this requires changing the same values in the `web/index.html`.
set INITIAL_MEMORY_BYTES=131072000
set MAX_MEMORY_BYTES=4294967296

call odin.exe build . -target:js_wasm32 -out:web/test.wasm -o:size -extra-linker-flags:"--export-table --import-memory --initial-memory=%INITIAL_MEMORY_BYTES% --max-memory=%MAX_MEMORY_BYTES%"

for /f "delims=" %%i in ('odin.exe root') do set "ODIN_ROOT=%%i"
copy "%ODIN_ROOT%\vendor\wgpu\wgpu.js" "web\wgpu.js"
copy "%ODIN_ROOT%\core\sys\wasm\js\odin.js" "web\odin.js"