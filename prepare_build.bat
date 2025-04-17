if not exist .\build\%1 mkdir .\build\%1
for /f "delims=" %%i in ('odin.exe root') do set "ODIN_ROOT=%%i"
copy "%ODIN_ROOT%\vendor\sdl2\sdl2.dll" .\build\%1