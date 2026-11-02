@echo off
rem Build the XNOR-popcount inference program with nvcc + MSVC.
rem Default target is the RTX 3060 (sm_86). Override with:  set CUDA_ARCH=sm_75 & build.bat
setlocal
if "%CUDA_ARCH%"=="" set CUDA_ARCH=sm_86

where cl >nul 2>nul
if not errorlevel 1 goto build
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -property installationPath`) do set "VSDIR=%%i"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul

:build
cd /d "%~dp0"
nvcc -O3 -std=c++17 -arch=%CUDA_ARCH% -allow-unsupported-compiler -o bnn_xnor.exe bnn_xnor.cu
exit /b %errorlevel%
