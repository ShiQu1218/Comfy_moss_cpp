@echo off
setlocal
set VSLANG=1033
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "MOSS_VS=%%i"
if not defined MOSS_VS exit /b 1
call "%MOSS_VS%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cmake -S "%~dp0." -B "%~dp0build-gpu" -G Ninja -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build "%~dp0build-gpu"

